import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import time
import gc
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from model.loss import Loss
from model.model_torth_new import DeCAF
import h5py
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from absl import app, flags
import imageio
from PIL import Image
from model.provider import DecafEndToEndProvider

NUM_GPUS = torch.cuda.device_count()

def save(model, directory, epoch=None, train_provider=None):
    if epoch is not None:
        directory = os.path.join(directory, "{}_model/".format(epoch))
    else:
        directory = os.path.join(directory, "latest/".format(epoch))
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, "model")
    if train_provider is not None:
        train_provider.save(directory)
    torch.save(model.state_dict(), path)
    print("saved to {}".format(path))
    return path

def restore(model,model_path):
    model.load_state_dict(torch.load(model_path))

def perm2RI(er, ei, n0):
    """
    Description: This function converts the recovered object's permittivity contrast into refractive index values more
                 commonly found in the literature.
    :param er:  Scalar, 2D, or 3D matrix containing object's real permittivity contrast
    :param ei:  scalar, 2D, or 3D matrix containing object's imaginary permittivity contrast
    :param n0:  Scalar value containing value of imaging medium's refractive index value. in Air n0 = 1, in water n0 = 1.33
    :return: nr: Scalar, 2D, or 3D matrix of object's real refractive index value.
             ni: scalar, 2D, or 3D matrix of object's imaginary refractive index value.
    """
    print("er max: {}, er min:{}".format(er.max(), er.min()))
    nr = np.sqrt(0.5 * ((n0**2 + er) + np.sqrt((n0**2 + er)**2 + ei**2)))
    ni = np.divide(ei, 2 * nr)
    return nr, ni

def predict(FLAGS, model, model_path, mesh_grid, Hreal=None, Himag=None):
    """Perform the inference of MLP
    Args:
        model_path: path to the saved model.
        mesh_grid:
        Hreal: phase light transfer function.
        Himag: absorption light transfer function.
    Returns:
        xhat: final reconstruction.
    """
    z, x, y, _ = mesh_grid.shape

    # placeholder
    xhat = np.zeros((z, x, y, 2))
    Hxhat = np.zeros((2, x, y))


    # Fourier transform
    F = lambda x: np.fft.fft2(x)
    iF = lambda x: np.fft.ifft2(x)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # self.to(device)

    #model=DeCAF(FLAGS)
    #model.to(device)
    # load model
    #model=restore(model,model_path)
    print('layer:',end=' ')
    # Start
    for start_layer in range(0, mesh_grid.shape[0], FLAGS.prediction_batch_size):
        print(start_layer, end=' ')
        # input mesh grid
        partial_mesh_grid = mesh_grid[
            start_layer : start_layer + FLAGS.prediction_batch_size
        ]
        reshaped_mesh_grid = torch.unsqueeze(
            torch.reshape(partial_mesh_grid, (-1, 3)), 0
        )
        # switch based on Hreal & Himag
        if Hreal is not None and Himag is not None:
            # extract H's
            partial_Hreal = Hreal[
                :, start_layer : start_layer + FLAGS.prediction_batch_size
            ]
            partial_Himag = Himag[
                :, start_layer : start_layer + FLAGS.prediction_batch_size
            ]
            partial_xhat = model(reshaped_mesh_grid[0], partial_mesh_grid, None, reuse=False,training=False)[1]
            Hxhat += torch.sum(
                torch.real(
                    iF(
                        torch.mul(partial_Hreal, F(partial_xhat[..., 0]))
                        + torch.mul(
                            partial_Himag, F(partial_xhat[..., 1])
                        )
                    )
                ),
                dim=1,
            )
        else:
            #print(start_layer)
            partial_xhat = model(reshaped_mesh_grid[0], partial_mesh_grid, None, reuse=False,training=False)[1]
        xhat[start_layer : start_layer + FLAGS.prediction_batch_size] = partial_xhat.cpu().detach().numpy()
    return Hxhat, xhat


def test(FLAGS,model):


    provider = DecafEndToEndProvider(h5py.File(FLAGS.input_dir, "r"), [0, 1])

    print("Inference started.")
    tic = time.perf_counter()

    rows = int(provider.measurement_size)
    cols = int(provider.measurement_size)

    assert FLAGS.z_min < FLAGS.z_min + FLAGS.z_delta < FLAGS.z_max
    key_zs = np.ceil((FLAGS.z_max + 1e-8 - FLAGS.z_min) / FLAGS.z_train_delta)
    zs = np.ceil((FLAGS.z_max + 1e-8 - FLAGS.z_min) / FLAGS.z_delta)

    if FLAGS.partial_render:
        scale = FLAGS.super_resolution_scale
        adjustment = 0.5 * (scale - 1) / scale
        rows_idx = np.linspace(
            FLAGS.row_render_min - adjustment,
            FLAGS.row_render_max - 1 + adjustment,
            num=int(
                (FLAGS.row_render_max - FLAGS.row_render_min)
                * FLAGS.super_resolution_scale
            ),
        )
        cols_idx = np.linspace(
            FLAGS.col_render_min - adjustment,
            FLAGS.col_render_max - 1 + adjustment,
            num=int(
                (FLAGS.col_render_max - FLAGS.col_render_min)
                * FLAGS.super_resolution_scale
            ),
        )

        assert FLAGS.z_min <= FLAGS.z_render_min <= FLAGS.z_render_max <= FLAGS.z_max

        key_z_min = (FLAGS.z_render_min - FLAGS.z_min) / FLAGS.z_train_delta
        partial_zs = np.ceil(
            (FLAGS.z_render_max + 1e-8 - FLAGS.z_render_min) / FLAGS.z_delta
        )
        key_z_max = (
                            FLAGS.z_render_min + (partial_zs - 1) * FLAGS.z_delta - FLAGS.z_min
                    ) / FLAGS.z_train_delta
        print(key_z_max)
        zs_idx = np.linspace(key_z_min, key_z_max, num=int(partial_zs))
    else:
        rows_idx = np.arange(0, rows)
        cols_idx = np.arange(0, cols)
        zs_idx = np.linspace(0, key_zs - 1, num=int(zs))
    r_mesh, z_mesh, c_mesh = np.meshgrid(cols_idx, zs_idx, rows_idx)

    r_mesh = (r_mesh / rows)[..., np.newaxis] - 0.5
    c_mesh = (c_mesh / cols)[..., np.newaxis] - 0.5
    z_mesh = (z_mesh / key_zs)[..., np.newaxis] - 0.5
    mesh_grid = np.concatenate((r_mesh, c_mesh, z_mesh), axis=-1) * 2
    mesh_grid = torch.tensor(mesh_grid).to(torch.float32).to('cuda').cuda()
    #FLAGS.view_size = rows_idx.size
    #model = Model()
    output_dir = FLAGS.result_save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    recon=predict(FLAGS,model,FLAGS.model_save_dir, mesh_grid)[1]
    if FLAGS.partial_render:
        save_name = "prediction_result_zmax{}_zmin{}_zdelta{}_{}_{}_{}_{}_{}_to_{}_x{}".format(
            output_dir,
            FLAGS.z_max,
            FLAGS.z_min,
            FLAGS.z_delta,
            FLAGS.row_render_min,
            FLAGS.row_render_max,
            FLAGS.col_render_min,
            FLAGS.col_render_max,
            FLAGS.z_render_min,
            FLAGS.z_render_max,
            FLAGS.super_resolution_scale,
        )
        save_path = "{}/{}.mat".format(output_dir, save_name)
    else:
        save_name = "prediction_result_zmax{}_zmin{}_zdelta{}".format(
            FLAGS.z_max,
            FLAGS.z_min,
            FLAGS.z_delta,
        )
        save_path = "{}/{}.mat".format(
            output_dir,
            save_name
        )
    toc = time.perf_counter()
    print("Inference ended in {:4} seconds.".format(toc - tic))
    with h5py.File(save_path, "w") as h5_file:
        h5_file.create_dataset("recon", data=recon)

    print("Prediction saved to {}".format(save_path))

    ab = recon[:, :, :, 1]
    ph = recon[:, :, :, 0]

    visual = "n_re"
    n_re, n_im = perm2RI(ph, ab, FLAGS.n0)
    result = n_re

    if visual == 'n_re':
        up = FLAGS.n0 + FLAGS.render_max;
        low = FLAGS.n0 + FLAGS.render_min;
    else:
        up = FLAGS.render_max
        low = FLAGS.render_min
    mu = (up + low) / 2;
    w = up - low
    print('low, up',low, up)
    print(np.min(result),np.max(result))
    result = np.clip(result, low, up)
    result -= np.min(result)
    result /= np.max(result)
    result *= 255
    result = result.astype(np.uint8)

    video_frames = []
    image_dir = '{}/{}/'.format(output_dir, save_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for idx, img in enumerate(result):
        #print(img)
        #print(img[0])
        im = Image.fromarray(img.T)
        #print(im)
        #print(im[0])
        im.save(image_dir + 'img_{}.png'.format(idx))
        video_frames.append(img.T)

    f = '{}/{}.mp4'.format(output_dir, save_name)
    imageio.mimwrite(f, video_frames, fps=8, quality=7)