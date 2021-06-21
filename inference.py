import time
import random
import math
import pprint
from tqdm import tqdm
import argparse
import os
from utils import resume_training, num_parameters, str2bool, generate_key, show_gpu_mem, load_checkpoint
from metrics import PSNR, L2
from dataloaders import SimpleDataloader
import torch

# settings
parser = argparse.ArgumentParser()
# IO parameters
parser.add_argument("--data_path", type=str, help="Path to the dir containing the testing dataset.")
parser.add_argument("--model_dir", type=str, help="The name of the experiment to be saved.", default=None)
# misc
parser.add_argument("--tqdm", type=str2bool, help="tqdm", default=True)
args = parser.parse_args()

# prepare torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# prepare dataloaders
loader = SimpleDataloader.basicImageDataloader(args.train_path,
                                               args.test_path,
                                               train_batch=args.train_batch,
                                               test_batch=args.test_batch,
                                               num_iters=args.num_iters,
                                               num_workers=args.num_workers)

# prepare experiment's logging files and checkpoint
experiment_dir = os.path.join(args.logdir, args.experiment)
starting_epoch, best_value_criterion, args = resume_training(ckpt_path=experiment_dir, args=args)

# prepare metrics and loss criterion
loss_fn = L2(boundary_ignore=0)
metrics_fn = {'l2': L2(), 'Psnr': PSNR()}

# prepare model and restore weights if experiment exists
from models.ResUnet import ResUNet as Net
model = Net(in_nc=3, out_nc=3, nb=1, nc=[32, 32, 32, 32], **vars(args)).to(device)

# multi gpu
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

load_checkpoint(os.path.join(experiment_dir, 'last.pth.tar'), model)

# ----------------------  perform training ----------------------
print(f'Starting training on : {device} \nTrainable params : {num_parameters(model)} \nSaving @ {experiment_dir}')
pprint.pprint(vars(args))

statistics_dict = {metric: 0. for metric in metrics_fn.keys()}  # initialize statistics
tic = time.time()
phase = 'test'
model.eval()  # Set model to evaluate mode

# Iterate over data.
num_iters = 0
for hr in tqdm(loader, disable=not args.tqdm):
    hr = hr.to(device)
    lr = hr
    # forward
    with torch.set_grad_enabled(False):
        pred = model(lr)

    # statistics
    vars_dict = {'pred': pred, 'gt': hr}
    current_stats = {key: metrics_fn[key](**vars_dict) for (key, value) in statistics_dict.items()}  # update all metrics
    statistics_dict = {key: value + current_stats['key'] for (key, value) in statistics_dict.items()}  # update all metrics

    # logging
    epoch_log = f'{num_iters:03d}'
    epoch_log += "".join([f' | {key}:{value:0.3f}' for key, value in current_stats.items()])  # display all metrics
    epoch_log += f' | gpu {show_gpu_mem()[1]:0.1f} Mb' if device != torch.device('cpu') else ''  # display gpu usage
    tqdm.write(epoch_log)

    num_iters += 1

# compute statistics
tac = time.time()
statistics_dict = {key: value / num_iters for (key, value) in statistics_dict.items()}
epoch_log = f'{num_iters:03d}'
epoch_log += "".join([f' | {key}:{value:0.3f}' for key, value in statistics_dict.items()])  # display all metrics
epoch_log += f' | gpu {show_gpu_mem()[1]:0.1f} Mb' if device != torch.device('cpu') else ''  # display gpu usage
tqdm.write(epoch_log)


