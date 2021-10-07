import os
import sys
import threading
import torch
from sacred import Experiment
from introvac.modules import utils
from os.path import join
from introvac import program
from introvac.modules.models import get_config
from pathlib import Path
import introvac.modules.log as log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ctx = threading.local()
ex = Experiment('Experiment name')
ctx.ex = ex


@ex.config
def cfg():
    logdir = 'results'
    model = 'Resnet'
    classifier = 'Logistic'
    # L2 regularization
    wd_class = 1E-5
    wd = 1E-5
    batch_size = 64
    dataset = 'CelebA'
    lr = 2E-4
    # Epochs to pretrain the IntroVAC without adversarial loss
    pretrain = 5
    lrs = [60, 90, 120]
    # Resnet channels [32, 64, 128, 256, 512]
    channels = [32, 64, 128, 256, 512]
    # Image resolution
    image_size = 128
    # Hidden dimension
    dim = 256
    output_dim = 2 * dim
    # Multiplication constant for learning rate schedule
    gamma = 0.5
    dataset_root = '.'
    logs_folder = join(logdir, 'logs')
    save_folder = join(logdir, 'save')
    epochs = 200
    log_freq = 100
    epoch_log_freq = 5
    # Attributes to classify
    attributes = ['FacialHair']
    # Ratio between images without and with attribute 2 for attribute Eyeglasses and 1 for attribute FacialHair
    ratio = 1
    # If multiple attributes are specified and binary it's true they are put together in a single
    # attribute
    binary = True
    gpu = 0
    # [class, reconst, kl, adversarial encoder, adversarial decoder]
    betas = [10.0, 100.0, 0.1, 0.01, 5]
    log_gradients = False


@ex.capture
def init(_log, save_folder, logs_folder):
    ctx.opt = utils.init_opt(ctx)
    included_opts = ['betas', 'dim', 'attributes', 'epochs']
    utils.build_filename(ctx, included_opts)
    save_folder = join(save_folder, ctx.opt['filename'])
    logs_folder = join(logs_folder, ctx.opt['filename'])
    ctx.opt['save_folder'] = save_folder
    ctx.device = torch.device(f"cuda:{ctx.opt['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    ctx.iter = 0
    ctx.epoch = 0

    try:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        _log.error("Cannot create output dir: A file exists with the same name specified in the path")
        sys.exit(1)
    log.attach_loggers([log.TfLogger], logs_folder)
    log.batch_log_freq = ctx.opt['log_freq']
    log.epoch_log_freq = ctx.opt['epoch_log_freq']


@ex.automain
def main():
    init()
    program.run(ctx)
