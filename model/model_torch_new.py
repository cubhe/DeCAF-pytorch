# DECAF training and predicting model with parallelization
# Created by Renhao Liu and Yu Sun, CIG, WUSTL, 2021

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

FLAGS = flags.FLAGS

NUM_Z = "nz"
INPUT_CHANNEL = "ic"
OUTPUT_CHANNEL = "oc"
MODEL_SCOPE = "infer_y"
NET_SCOPE = "MLP"
DNCNN_SCOPE = "DnCNN"

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


class DeCAF(nn.Module):
    def __init__(self, FLAGS, net_kargs=None, name="model_summary"):
        super(DeCAF, self).__init__()
        # Setup parameters
        self.name = name
        self.tf_summary_dir = "{}/{}".format(FLAGS.tf_summary_dir, name)
        if net_kargs is None:
            self.net_kargs = {
                "skip_layers": FLAGS.mlp_skip_layer,
                "mlp_layer_num": FLAGS.mlp_layer_num,
                "kernel_size": FLAGS.mlp_kernel_size,
                "L_xy": FLAGS.xy_encoding_num,
                "L_z": FLAGS.z_encoding_num,
            }
        else:
            self.net_kargs = net_kargs
        print('FLAGS.mlp_skip_layer', FLAGS.mlp_skip_layer)
        input_dim = FLAGS.xy_encoding_num * 24 + FLAGS.z_encoding_num * 2
        output_dim = 2
        FLAGS.mlp_kernel_size = 200
        # FLAGS.mlp_kernel_size=208
        self.inputlayer = nn.Linear(input_dim, FLAGS.mlp_kernel_size)
        self.skiplayer = nn.Linear(FLAGS.mlp_kernel_size + input_dim, FLAGS.mlp_kernel_size)
        FLAGS.mlp_layer_num = 10
        self.lineares = nn.ModuleList(
            [nn.Linear(FLAGS.mlp_kernel_size, FLAGS.mlp_kernel_size) for i in range(FLAGS.mlp_layer_num)])
        self.outputlayer = nn.Linear(FLAGS.mlp_kernel_size, output_dim)
        self.le_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.sigmoid = nn.Sigmoid()

    ###########################
    ###     Neural Nets     ###
    ###########################

    def forward(self, coordinates, Hreal, Himag, padding=None, mask=None, reuse=False, training=True, epochs=0):
        self.training = training
        # MLP network
        xhat = self.__neural_repres(coordinates, Hreal.shape, **self.net_kargs)
        if self.training:
            mask = mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
            # 调试 debug
            Hxhat = self.__forward_op(xhat * mask, Hreal, Himag, padding)

        else:
            Hxhat = self.__forward_op(xhat, Hreal, Himag, padding)
        self.epochs = epochs
        # Hxhat=(F.normalize(Hxhat,dim=0))
        # Hxhat=(Hxhat/Hxhat.max()+1)/2
        return Hxhat, xhat

    def __neural_repres(self, in_node, x_shape, skip_layers=[], mlp_layer_num=10, kernel_size=256, L_xy=6, L_z=5):
        # positional encoding
        if FLAGS.positional_encoding_type == "exp_diag":
            s = torch.sin(torch.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[
                :, None
                ]
            c = torch.cos(torch.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[
                :, None
                ]
            fourier_mapping = torch.cat((s, c), dim=1).T
            fourier_mapping = fourier_mapping.to('cuda').cuda()
            xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)
            # print('xy_freq',xy_freq.shape)
            for l in range(L_xy):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * xy_freq),
                        torch.cos(2 ** l * np.pi * xy_freq),
                    ],
                    dim=-1,
                )
                if l == 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
                # print('cur_freq',cur_freq.shape)
            # print('tot_freq',tot_freq.shape)
            for l in range(L_z):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                        torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                    ],
                    dim=-1,
                )
                # print('in_node[:, 2].unsqueeze(-1)',in_node[:, 2].unsqueeze(-1).shape)
                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
                # print('cur_freq',cur_freq.shape)
            # print('tot_freq',tot_freq.shape)
        elif FLAGS.positional_encoding_type == 'exp':
            for l in range(L_xy):  # fourier feature map
                indicator = torch.tensor([1., 1., 1. if l < L_z else 0.])
                cur_freq = torch.cat([torch.sin(indicator * 2 ** l * np.pi * in_node),
                                      torch.cos(indicator * 2 ** l * np.pi * in_node)], dim=-1)
                if l is 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
        elif FLAGS.positional_encoding_type == 'fourier_fixed_xy':
            torch.manual_seed(10)
            fourier_mapping = torch.normal(0, FLAGS.sig_xy, (FLAGS.fourier_encoding_size, 2)).float()

            xy_freq = torch.matmul(in_node[:, :2], fourier_mapping.T)
            xy_freq = torch.cat([torch.sin(2 * np.pi * xy_freq),
                                 torch.cos(2 * np.pi * xy_freq)], dim=-1)

            tot_freq = xy_freq
            for l in range(L_z):
                cur_freq = torch.cat([torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                                      torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))], dim=-1)
                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
        else:
            raise NotImplementedError(FLAGS.positional_encoding_type)

        # input to MLP
        in_node = tot_freq
        x = self.inputlayer(in_node)
        x = self.le_relu(x)
        # print(x.shape)
        # in_node = torch.cat([x, tot_freq], -1)
        # print(in_node.shape)
        layer_cout = 1
        # print(skip_layers)
        for f in self.lineares:
            layer_cout += 1
            if layer_cout in skip_layers:
                x = torch.cat([x, tot_freq], -1)
                x = self.skiplayer(x)
                x = F.leaky_relu(x, negative_slope=0.2)
            x = f(x)
            x = F.leaky_relu(x, negative_slope=0.2)
        output = self.outputlayer(x)
        # print(output.max(),output.min())
        # output = (F.sigmoid(output)*2-1)
        # print(output.max(),output.min())
        # input encoder
        # if FLAGS.task_type == "aidt":
        #     kernel_initializer = None
        # elif FLAGS.task_type == "idt":
        #     kernel_initializer = nn.init.uniform_(-0.05, 0.05)
        # elif FLAGS.task_type == "midt":
        #     kernel_initializer = None
        # else:
        #     raise NotImplementedError

        # print(FLAGS.output_scale)
        # print(output.max(),output.min())
        # print(output.min())
        output = output / FLAGS.output_scale
        # output=self.sigmoid(output)/100
        # output = F.normalize(output,dim=0)
        # reshape output to x
        if self.training:
            xhat = output.view(x_shape[1], x_shape[2], FLAGS.view_size, FLAGS.view_size,
                               2)  # [1, Z, X, Y, Real/Imagenary]
        else:
            xhat = output.view(1, x_shape[0], x_shape[1], x_shape[2], 2)  # [1, Z, X, Y, Real/Imagenary]
        return xhat

    def __forward_op(self, x, Hreal, Himag, padding):
        if self.training:
            padding = tuple(padding)
            # print(padding)
            # print(padding[3][0])
            # print('padding')
            pad_step = 20
            padded_field = F.pad(x, (0, 0, padding[3][0], padding[3][0], padding[3][0], padding[3][0]))  # orign
            # print(padded_field.shape)
            # padded_field = F.pad(padded_field, (0, 0, pad_step, pad_step, pad_step, pad_step))
            # print(padded_field.shape)
            padded_phase = padded_field[:, :, :, :, 0]
            padded_absorption = padded_field[:, :, :, :, 1]
            # print(Hreal.shape)
            # Hreal_pad=F.pad(Hreal, (0, 0, 0,0,pad_step, pad_step, pad_step, pad_step))
            # print(Hreal_pad.shape)
            # #调试关闭 运行记得打开
            # padded_phase_new = torch.stack([padded_phase for i in range(Hreal.shape[0])])
            # padded_absorption_new = torch.stack([padded_absorption for i in range(Himag.shape[0])])
            transferred_field = torch.fft.ifft2(
                torch.mul(Hreal, torch.fft.fft2(padded_phase.expand(Hreal.shape).to(torch.complex64)))
                + torch.mul(Himag, torch.fft.fft2(padded_absorption.expand(Himag.shape).to(torch.complex64)))
            )
            # transferred_field = torch.fft.ifft2(
            #     torch.mul(Hreal, torch.fft.fft2(padded_phase_new.to(torch.complex64)))
            #     + torch.mul(Himag, torch.fft.fft2(padded_absorption_new.to(torch.complex64)))
            # )
            Hxhat = torch.sum(torch.real(transferred_field), dim=(1, 2))

            return Hxhat
        else:
            return x

    def save(self, directory, epoch=None, train_provider=None):
        if epoch is not None:
            directory = os.path.join(directory, "{}_model/".format(epoch))
        else:
            directory = os.path.join(directory, "latest/".format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "model")
        if train_provider is not None:
            train_provider.save(directory)
        torch.save(self.state_dict(), path)
        print("saved to {}".format(path))
        return path

    def restore(self, model_path):

        param = torch.load(model_path)
        # param_model=self.state_dict()
        # new_dict={}
        # for k,v in param.items():
        #     if k in param_model:
        #         print(k)
        #         print(v)
        self.load_state_dict(param, strict=False)

        # self.load_state_dict(torch.load(model_path),strict=False)

    ##############################
    ###     Loss Functions     ###
    ##############################

    def __tower_loss(self, tower_idx, Hreal, Himag, reuse=False):
        # get input coordinates & measurements & padding
        x = self.Xs[tower_idx, ...]
        y = self.Ys[tower_idx, ...]
        padding = self.Ps[tower_idx, ...]
        mask = self.Ms[tower_idx, ...]

        # inference
        Hxhat, xhat = self.inference(x, Hreal, Himag, padding, mask, reuse=reuse)
        # data fidelity
        if FLAGS.loss == "l1":
            mse = torch.mean(torch.abs(Hxhat - y))
        elif FLAGS.loss == "l2":
            mse = torch.mean(torch.square(Hxhat - y)) / 2
        else:
            raise NotImplementedError
        # regularizer
        if FLAGS.regularize_type == "dncnn2d":
            xhat_trans = torch.transpose(
                torch.squeeze(xhat), 3, 0
            )  # [1, Z, X, Y, Real/Imagenary]
            xhat_concat = torch.cat([xhat_trans[0, ...], xhat_trans[1, ...]], 0)
            xhat_expand = xhat_concat.unsqueeze(3)
            phase_regularize_value = self.__dncnn_2d(xhat_expand, reuse=reuse)
            absorption_regularize_value = torch.tensor(0.0)
        else:
            raise NotImplementedError

        if FLAGS.tv3d_z_reg_weight != 0:
            tv_z = self.__total_variation_z(xhat[..., 0])
            tv_z += self.__total_variation_z(xhat[..., 1])
        else:
            tv_z = torch.tensor(0.0)

        # final loss
        loss = (
                mse
                + FLAGS.regularize_weight
                * (absorption_regularize_value + phase_regularize_value)
                + FLAGS.tv3d_z_reg_weight * tv_z
        )

        return (
            loss,
            mse,
            phase_regularize_value,
            absorption_regularize_value,
            xhat,
            Hxhat,
            y,
        )

    def __total_variation_2d(self, images):
        pixel_dif2 = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = torch.sum(pixel_dif2) + torch.sum(pixel_dif3)
        return total_var

    def __total_variation_z(self, images):
        """
        Normalized total variation 3d
        :param images: Images should have 4 dims: batch_size, z, x, y
        :return:
        """
        pixel_dif1 = torch.abs(images[:, 1:, :, :] - images[:, :-1, :, :])
        total_var = torch.sum(pixel_dif1)
        return total_var

    def __dncnn_inference(
            self,
            input,
            reuse,
            output_channel=1,
            layer_num=10,
            filter_size=3,
            feature_root=64,
    ):
        # input layer
        with torch.no_grad():
            in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size // 2)
            in_node = F.relu(in_node(input))
            # composite convolutional layers
            for layer in range(2, layer_num):
                in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False)
                in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
            # output layer and residual learning
            in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)
            output = input - in_node
        return output

    def __dncnn_2d(self, images):  # [N, H, W, C]
        """
        DnCNN as 2.5 dimensional denoiser based on l-2 norm
        """
        a_min = FLAGS.DnCNN_normalization_min
        a_max = FLAGS.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1))
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized ** 2)
        return dncnn_res





