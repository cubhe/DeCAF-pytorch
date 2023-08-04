"""
Training main function for DECAF.
Created by Renhao Liu, CIG, WUSTL, 2021.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import logging
import torch

#import tensorflow as tf
#tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import h5py
from absl import app
from absl import flags
import time

#from model.model import Model
from model.provider import DecafEndToEndProvider
from train import train
from test  import predict


FLAGS = flags.FLAGS

# Version parameters
flags.DEFINE_string("name", "DECAF", "model name")

# Replace current parameters.
flags.DEFINE_string("input_dir", "", "input_file")
flags.DEFINE_string("model_save_dir", "saved_model", "directory for saving model")

# Training hyper parameters
flags.DEFINE_integer("epochs", 40000, "number of training epochs")
flags.DEFINE_integer("iters_per_epoch", 200, "number of training epochs")
flags.DEFINE_integer("test_step", 200, "number of training epochs")
flags.DEFINE_float("start_lr", 1e-4, "number of training epochs")
flags.DEFINE_float("end_lr", 5e-5, "number of training epochs")


# Model parameters
flags.DEFINE_string("tf_summary_dir", "log", "directory for tf summary log")
flags.DEFINE_enum(
    "positional_encoding_type", "exp_diag", ["exp_diag", "exp", "fourier_fixed_xy"], "positional_encoding_type", )
flags.DEFINE_float("dia_digree", 45, "degrees per each encoding in exp_diag")
flags.DEFINE_enum(
    "mlp_activation", "leaky_relu", ["leaky_relu"], "Activation functions for mlp", )
flags.DEFINE_integer("mlp_layer_num", 8, "number of layers in mlp network")
flags.DEFINE_integer("mlp_kernel_size", 208, "width of mlp")
flags.DEFINE_integer('fourier_encoding_size', 256, "number of rows in fourier matrix")
flags.DEFINE_float("sig_xy", 26.0, "Fourier encoding sig_xy")
flags.DEFINE_float("sig_z", 1.0, "Fourier encoding sig_z")
flags.DEFINE_integer(
    "xy_encoding_num", 6, "number of frequecncies expanded in the spatial dimensions"
)
flags.DEFINE_integer(
    "z_encoding_num", 5, "number of frequecncies expanded in the depth dimension"
)
flags.DEFINE_multi_integer("mlp_skip_layer", [5], "skip layers in the mlp network")
flags.DEFINE_float("output_scale", 5, "neural network out put scale")

# Regularization parameters
flags.DEFINE_enum("regularize_type", "dncnn2d", ["dncnn2d"], "type of the network", )
flags.DEFINE_float("regularize_weight", 0.0, "Weight for regularizer")
flags.DEFINE_float(
    "tv3d_z_reg_weight", 3.079699, "Reg weight scaling for z axis in 3dtv"
)
flags.DEFINE_string(
    "DnCNN_model_path",
    "/export/project/sun.yu/projects/DnCNN/cnn_trained/DnCNN_sigma=5.0/models/final/model",
    "model path of pre-trained DnCNN",
)
flags.DEFINE_float("DnCNN_normalization_min", -0.05, "DnCNN normalization min")
flags.DEFINE_float("DnCNN_normalization_max", 0.05, "DnCNN normalization max")

# Training parameter
flags.DEFINE_integer("start_epoch", 0, "start epoch, useful for continue training")
# flags.DEFINE_integer("iters_per_epoch", 300, "num of iters for each resampling")
flags.DEFINE_integer("image_save_epoch", 5000, "number of iteration to save one image")
flags.DEFINE_integer(
    "intermediate_result_save_epoch",
    100,
    "number of iterations to save intermediate result",
)
flags.DEFINE_integer("log_iter", 25, "number of iteration to log to console")
flags.DEFINE_integer("model_save_epoch", 5000, "epoch per intermediate model")
flags.DEFINE_integer(
    "num_measurements_per_batch",
    -1,
    "number of measurements per batch. negative value for all measurements",
)

# Prediction parameters
flags.DEFINE_integer("prediction_batch_size", 1, "Batch size for prediction")


# test configures
flags.DEFINE_string("result_save_dir", "result", "directory for saving results")
# Prediction config.
flags.DEFINE_float("z_min", -10, "minimum depth in micrometer")
flags.DEFINE_float("z_max", 16, "maximum depth in micrometer")
flags.DEFINE_float("z_train_delta", 0.5, "z delta in training data")
flags.DEFINE_float("z_delta", 0.1, "depth for each layer in micrometer")

flags.DEFINE_boolean(
    "partial_render",
    False,
    "Whether to render a subset of z. z_render_min, z_render_max only"
    "works if this is True",
)
flags.DEFINE_float("z_render_min", -20, "minimum depth to render in micrometer")
flags.DEFINE_float("z_render_max", 60, "maximum depth to render in micrometer")

flags.DEFINE_integer("row_render_min", 0, "minimum row to render in pixel")
flags.DEFINE_integer("row_render_max", 100, "maximum row to render in pixel")

flags.DEFINE_integer("col_render_min", 0, "minimum col to render in pixel")
flags.DEFINE_integer("col_render_max", 100, "maximum col to render in pixel")
flags.DEFINE_float("super_resolution_scale", 1, "super resolution scale")

# Render config.
flags.DEFINE_float("n0", 1.33, "n0 of the medium.")
flags.DEFINE_float("render_max", 0.02, "Range above average in rendering.")
flags.DEFINE_float("render_min", 0.02, "Range below average in rendering.")



def main(argv):
    """
        DECAF main function
    """
    lr_start = FLAGS.start_lr
    lr_end = FLAGS.end_lr
    epochs = FLAGS.epochs
    multiplier = (lr_end / lr_start) ** (1 / epochs)
    decayed_lr = [lr_start * (multiplier ** x) for x in range(epochs)]

    train_kargs = {
        "epochs": epochs,
        "learning_rate": decayed_lr,
    }
    directory = FLAGS.model_save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = open(directory + "/config.txt", "w")
    config_file.write(FLAGS.flags_into_string())
    config_file.close()

    #FLAGS.input_dir = r'/home/cubhe/桌面/' + FLAGS.input_dir
    h5_file = h5py.File(FLAGS.input_dir, "r")

    train_provider = DecafEndToEndProvider(h5_file)
    train(FLAGS, train_provider, learning_rate=decayed_lr, epochs=epochs,restore=None)
    #model = Model(name=FLAGS.name)
    #model.train(FLAGS,FLAGS.model_save_dir, train_provider, **train_kargs)

if __name__ == "__main__":
    app.run(main)
