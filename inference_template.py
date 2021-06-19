import time
import random
from tqdm import tqdm
import argparse
from utils import resume_training,num_parameters,save_checkpoint, generate_key, show_gpu_mem
import os
import numpy as np
import torch
from models.mfsr import Net
from metrics import PSNR,L2
import dataloaders.dataloader as dataloader

#settings
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=2e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr_decay", type=float, help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--num_epochs", type=int, help="Total number of epochs to train", default=250)
parser.add_argument("--model_dir", type=str, help="Results' dir path", default='trained_models')
parser.add_argument("--experiment", type=str, help="The name of the experiment to be saved.", default=None)
parser.add_argument("--data_path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/")
args = parser.parse_args()


#fixed random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

#prepare experiment
experiment = args.model_name if args.model_name is not None else generate_key()
ckpt_path = args.ckpt_path if args.ckpt_path is not None else os.path.join(args.model_dir, experiment,'model.cpkt')
experiment_dir =  os.path.dirname(ckpt_path)

#check devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

#prepare data
loaders = dataloader.basicImageDataloader(args.train_path, args.test_path)

#prepare model and restore weights
model = Net(**args).to(device)
if args.n_gpu >1:
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
starting_epoch, best_psnr = resume_training(ckpt_path,model,optimizer, args, inference=True)

#perform inference
print(f'Starting infernce on {device} trainable params {num_parameters(model)} saving @ {ckpt_path}')
model.eval()  # Set model to evaluate mode


