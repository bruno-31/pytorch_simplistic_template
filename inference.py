import time
import random
from tqdm import tqdm
import argparse
from utils import resume_training,num_parameters,str2bool,save_checkpoint, generate_key, show_gpu_mem, seed_worker
import os

from metrics import PSNR,L2
from dataloaders import dataloader

import numpy as np
import torch

#settings
parser = argparse.ArgumentParser()

# IO parameters
parser.add_argument("--train_path", type=str, help="Path to the dir containing the training dataset.")
parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing dataset.")
parser.add_argument("--logdir", type=str, help="dir to save experiments", default='experiments_logs')
parser.add_argument("--experiment", type=str, help="The name of the experiment to be saved.", default=None)
parser.add_argument("--tensorboard", type=str2bool, help="The name of the experiment to be saved.", default=False)
# optimization parameters
parser.add_argument("--lr", type=float, dest="lr", help="Learning rate", default=2e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr_decay", type=float, help="Learning rate decay (on step)", default=0.35)
parser.add_argument("--num_epochs", type=int, help="Total number of epochs to train", default=250)
parser.add_argument("--num_iters", type=int, help="Num iters per epoch", default=4000)
# misc
parser.add_argument("--n_gpu", type=int, help="Learning rate decrease step", default=1)
parser.add_argument("--num_workers", type=int, help="Learning rate decrease step", default=1)
parser.add_argument("--deterministic", type=str2bool, help="Learning rate decrease step", default=False)
parser.add_argument("--test_only", type=str2bool, help="Only testing", default=False)
parser.add_argument("--tqdm", type=str2bool, help="tqdm", default=True)
args = parser.parse_args()


# prepare torch device and rnd seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if args.deterministic:
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    generator = torch.Generator(); generator.manual_seed(0)  # fixing Dataloaders workers rnd seeds
    torch.use_deterministic_algorithms(True)

# prepare experiment's logging files and checkpoint
experiment = args.experiment if args.experiment is not None else generate_key()
ckpt_path = os.path.join(args.logdir, experiment,'model.cpkt')
experiment_dir =  os.path.dirname(ckpt_path)
os.makedirs(experiment_dir, exist_ok=True)
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(experiment_dir) # TODO write on same path ?

# prepare dataloaders
from datasets import ImageDataset
loaders_dict = dataloader.basicImageDataloader(args.train_path,
                                               args.test_path,
                                               worker_init_fn=seed_worker if args.deterministic else None,
                                               generator=generator if args.deterministic else None,
                                               num_iters=args.num_iters,
                                               num_workers=args.num_workers)

# prepare metrics and loss criterion
loss_fn = L2(boundary_ignore=0)
metrics_fn = {'l2':L2(), 'Psnr':PSNR()}
criterion = 'Psnr' # key of the reference metric for best model saving

# prepare model and restore weights if experiment exists
from models.ResUnet import ResUNet as Net
model = Net(in_nc=3,out_nc=3, **vars(args)).to(device)
if args.n_gpu >1:
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
starting_epoch, best_value_criterion = resume_training(ckpt_path, model, optimizer) # check existing checkpoints

# ----------------------  perform training ----------------------
print(f'Starting training on {device} trainable params {num_parameters(model)} saving @ {ckpt_path} '
      f'\n parameters {vars(args)}')
for epoch in tqdm(range(starting_epoch, args.num_epochs)):

    #init statistics
    is_best = False
    statistics_dict = {metric:0. for metric in metrics_fn.keys()} # initialize statistics
    tic = time.time()

    for phase in ['test'] if args.test_only else ['train','test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # Iterate over data.
        num_iters = 0
        for hr in tqdm(loaders_dict[phase], disable=not args.tqdm):
            hr = hr.to(device)
            lr = hr
            # forward
            with torch.set_grad_enabled(phase == 'train'):
                pred = model(lr)
                loss = loss_fn(pred, hr)
                vars = {'pred':pred, 'gt':hr}

                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad() # zero the parameter gradients
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # update statistics
            statistics_dict = {key: value + metrics_fn[key](**vars) for (key, value) in statistics_dict.items()} # update all metrics
            num_iters += 1

        # logging metrics
        tac = time.time()
        statistics_dict = {key: value / num_iters for (key, value) in statistics_dict.items()}
        # console logging
        epoch_log = f'{phase} {epoch} | {(tac-tic):0.2f}s ({(tac-tic)/num_iters:0.2f}s/iter) '
        epoch_log += "".join([f' | {key}:{value:0.3f}' for key,value in statistics_dict.items()]) # display all metrics
        epoch_log = epoch_log + f' | gpu {show_gpu_mem()[1]:0.1f} Mb' if device!=torch.device('cpu') else epoch_log # display gpu usage
        tqdm.write(epoch_log)
        # tensorboard loggin
        if args.tensorboard:
            [writer.add_scalar(f'{key}/{phase}', value, epoch) for (key, value) in statistics_dict.items()]
            writer.flush()

        #saving best module according to chosed criterion
        if phase == 'test':
            if statistics_dict[criterion] > best_value_criterion:
                best_value_criterion = statistics_dict[criterion] ; is_best = True

    # deep copy the model
    if not args.test_only:
        save_checkpoint(model, optimizer, args, epoch, best_value_criterion, is_best, ckpt_path)

if args.tensorboard:
    writer.close()