# DECAF training and predicting model with parallelization
# Created by Renzhi He, UC Davis, 2023

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import skimage
from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
from absl import flags
import logging

from .ssim import SSIM
from .dncnn import DnCNN

# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()


########################################
###       Tensorboard & Helper       ###
########################################

def record_summary(writer, name, value, step):
    writer.add_scalar(name, value, step)
    writer.flush()


def reshape_image(image):
    if len(image.shape) == 2:
        image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    elif len(image.shape) == 3:
        image_reshaped = image.unsqueeze(-1)
    else:
        image_reshaped = image
    return image_reshaped


def reshape_image_2(image):
    image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    return image_reshaped


def reshape_image_3(image):
    image_reshaped = image.unsqueeze(-1)
    return image_reshaped


def reshape_image_5(image):
    shape = image.shape
    image_reshaped = image.view(-1, shape[2], shape[3], 1)
    return image_reshaped


#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == '__main__':
    main()


class Loss(nn.Module):
    def __init__(self, DnCNNN_channels=1, tower_idx=None, Hreal=None, Himag=None):
        super(Loss, self).__init__()
        self.tower_idx = tower_idx
        self.Hreal = Hreal
        self.Himag = Himag
        self.SSIM = SSIM()
        self.TVLoss = TVLoss()

        # dncnn
        num_of_layers = 17
        logdir = "./model/dncnn_logs/DnCNN-S-25"
        net = DnCNN(channels=DnCNNN_channels, num_of_layers=num_of_layers)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(os.path.join(logdir, 'net.pth')), strict=False)
        # print('load DnCNN Done...')
        # lock the parameters of DnCNN
        for name, parameter in model.named_parameters():
            parameter.requires_grad = False
        self.dncnn = model

        # Setup parameters

    ##############################
    ###     Loss Functions     ###
    ##############################

    def forward(self, FLAGS, Hxhat, xhat, y, steps, tower_idx=0, reuse=False):
        # get input coordinates & measurements & padding
        # x = self.Xs[tower_idx, ...]
        # y = self.Ys[tower_idx, ...]
        # padding = self.Ps[tower_idx, ...]
        # mask = self.Ms[tower_idx, ...]

        # inference
        # FLAGS.loss = "l2"
        # print(FLAGS.loss)
        # data fidelity
        if FLAGS.loss == "l1":
            mse = torch.mean(torch.abs(Hxhat - y)) / 10
        elif FLAGS.loss == "l2":
            mse = torch.mean(torch.square(Hxhat - y)) / 2
        else:
            raise NotImplementedError

        # regularizer

        if FLAGS.regularize_type == "dncnn2d":
            # print(xhat.shape)
            # print(xhat.grad_fn)
            xhat_trans = torch.transpose(torch.squeeze(xhat), 3, 0)
            xhat_concat = torch.cat([xhat_trans[0, ...], xhat_trans[1, ...]], 2)
            xhat_concat = torch.transpose(xhat_concat, 2, 0)
            xhat_expand = xhat_concat.unsqueeze(1)
            with torch.no_grad():
                dncnn_loss = self.dncnn(xhat_expand)
            phase_regularize_value = (dncnn_loss.mean().squeeze()) * 1
            # phase_regularize_value = dncnn_loss(FLAGS, xhat_expand.to('cpu'), reuse=reuse)

            # 记得打开
            # phase_regularize_value = torch.tensor(0.0)
            absorption_regularize_value = torch.tensor(0.0)
        else:
            raise NotImplementedError

        # print(y.shape)
        # print(Hxhat.shape)
        # y_trans = torch.transpose(torch.squeeze(y), 3, 0)  # [1, Z, X, Y, Real/Imagenary]
        # y_concat = torch.cat([y_trans[0, ...], y_trans[1, ...]], 2)
        # y_concat = torch.transpose(y_concat, 2, 0)
        # y_expand = y_concat.unsqueeze(1)
        Hxhat = Hxhat.unsqueeze(1)
        y = y.unsqueeze(1)
        # print(y.shape)
        # print(Hxhat.shape)
        ssim = (1 - self.SSIM(Hxhat, y)) / 2
        # print(ssim)

        if FLAGS.tv3d_z_reg_weight != 0:
            tv_z = self.__total_variation_z(xhat[..., 0])
            tv_z += self.__total_variation_z(xhat[..., 1])
        else:
            tv_z = torch.tensor(0.0)
        # print(FLAGS.tv3d_z_reg_weight*tv_z)
        tv_xy = self.TVLoss(xhat_expand)
        # FLAGS.tv3d_z_reg_weight=0.00005
        # mse=mse*0.1

        if steps < 9600:
            ratio_mse = 1.0
            ratio_ssim = 0.0
            ratio_tv_z = 3e-07
            ratio_tv_xy = 0
            ratio_reg = 1e-05
        elif steps < 19200:
            ratio_mse = 1
            ratio_ssim = 0.01
            ratio_tv_z = 8e-07
            ratio_tv_xy = 1e-06
            ratio_reg = 5e-03
        else:
            FLAGS.loss = "l1"
            ratio_mse = 1
            ratio_ssim = 0.05
            ratio_tv_z = 8e-07
            ratio_tv_xy = 1e-06
            ratio_reg = 5e-03

        # print(steps)
        mse = mse * ratio_mse
        ssim = ssim * ratio_ssim
        tv_z = tv_z * ratio_tv_z
        tv_xy = tv_xy * ratio_tv_xy
        phase_regularize_value = phase_regularize_value * ratio_reg
        ##

        # --regularize_weight=3e-07
        # --tv3d_z_reg_weight=1.5e-07

        # final loss
        loss = (
                mse
                + ssim
                + tv_xy
                + (absorption_regularize_value + phase_regularize_value)
                + tv_z
        )

        return (
            loss,
            mse,
            phase_regularize_value,
            absorption_regularize_value,
            tv_z,
            ssim,
            tv_xy,
            # xhat,
            # Hxhat,
            # y,
        )

    def __total_variation_2d(self, images):
        pixel_dif2 = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = torch.sum(pixel_dif2) + torch.sum(pixel_dif3)
        return total_var

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def __total_variation_z(self, images):
        """
        Normalized total variation 3d
        :param images: Images should have 4 dims: batch_size, z, x, y
        :return:
        """
        pixel_dif1 = torch.abs(images[:, 1:, :, :] - images[:, :-1, :, :])
        total_var = torch.sum(pixel_dif1)
        return total_var
    # def __dncnn_inference(
    #     self,
    #     input,
    #     reuse,
    #     output_channel=1,
    #     layer_num=10,
    #     filter_size=3,
    #     feature_root=64,
    # ):
    #     # input layer
    #     with torch.no_grad():
    #         in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size//2)
    #         in_node = F.relu(in_node)
    #         # composite convolutional layers
    #         for layer in range(2, layer_num):
    #             in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size//2, bias=False)
    #             in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
    #         # output layer and residual learning
    #         in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size//2)
    #         output = input - in_node
    #     return output

    # def __dncnn_2d(self, FLAGS, images,reuse=True):  # [N, H, W, C]
    #     """
    #     DnCNN as 2.5 dimensional denoiser based on l-2 norm
    #     """
    #     a_min = FLAGS.DnCNN_normalization_min
    #     a_max = FLAGS.DnCNN_normalization_max
    #     normalized = (images - a_min) / (a_max - a_min)
    #     denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1),reuse)
    #     denormalized = denoised * (a_max - a_min) + a_min
    #     dncnn_res = torch.sum(denormalized**2)
    #     return dncnn_res


class dncnn_2d(nn.Module):
    def __init__(self, FLAGS, input_channel, output_channel=1, layer_num=10, filter_size=3, feature_root=64):
        super(dncnn_2d, self).__init__()
        self.input_conv = nn.Conv2d(input_channel, feature_root, filter_size, padding=filter_size // 2)
        self.convs = nn.ModuleList([
            nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False) for i in
            range(layer_num)
        ])
        self.output_conv = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)
        self.relu = nn.ReLU()

        # in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size // 2)
        # in_node = F.relu(in_node)
        # # composite convolutional layers
        # for layer in range(2, layer_num):
        #     in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False)
        #     in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
        # # output layer and residual learning
        # in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)

    def forward(self, FLAGS, images, reuse=True):
        a_min = FLAGS.DnCNN_normalization_min
        a_max = FLAGS.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1), reuse)
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized ** 2)
        return dncnn_res

        return 0

    def __dncnn_inference(self, input, reuse=True):
        x = self.input_conv(input)
        for f in self.convs:
            x = f(x)
            x = self.relu(x)
        output = self.output_conv(x)

        return output