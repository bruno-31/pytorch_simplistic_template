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

parser.add_argument("--train_path", type=str, help="Path to the dir containing the training dataset.")
parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing dataset.")

parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=2e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr_decay", type=float, help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--num_epochs", type=int, help="Total number of epochs to train", default=250)
parser.add_argument("--logdir", type=str, help="dir to save experiments", default='experiments_logs')
parser.add_argument("--experiment", type=str, help="The name of the experiment to be saved.", default=None)
parser.add_argument("--tensorboard", type=str2bool, help="The name of the experiment to be saved.", default=False)
parser.add_argument("--n_gpu", type=int, help="Learning rate decrease step", default=1)
parser.add_argument("--num_workers", type=int, help="Learning rate decrease step", default=1)
parser.add_argument("--deterministic", type=str2bool, help="Learning rate decrease step", default=False)
args = parser.parse_args()


# fixed random seeds for reproducibility
if args.deterministic:
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    generator = torch.Generator(); generator.manual_seed(0) # fixing Dataloaders workers rnd seeds
    torch.use_deterministic_algorithms(True)


# prepare experiment logging files and checkpoint and generate experiment id
experiment = args.model_name if args.model_name is not None else generate_key()
ckpt_path = args.ckpt_path if args.ckpt_path is not None else os.path.join(args.model_dir, experiment,'model.cpkt')
experiment_dir =  os.path.dirname(ckpt_path)
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(experiment_dir) # TODO write on same path ?

# prepare torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# prepare dataloaders
loaders = dataloader.basicImageDataloader(args.train_path,
                                          args.test_path,
                                          worker_init_fn=seed_worker if args.deterministic else None,
                                          generator=generator if args.deterministic else None,
                                          num_workers=args.num_workers)

# prepare model and restore weights
from models.ResUnet import ResUNet as Net
model = Net(**args).to(device)
if args.n_gpu >1:
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
starting_epoch, best_psnr = resume_training(ckpt_path,model,optimizer, args)

# prepare metrics
Loss = L2(boundary_ignore=0)
metrics = {'l2':L2(), 'Psnr':PSNR()}

# /// perform training ///
print(f'Starting training on {device} trainable params {num_parameters(model)} saving @ {ckpt_path}')
for epoch in tqdm(range(starting_epoch, args.num_epochs)):

    #init statistics
    psnr_epoch = loss_epoch = geom_error = 0
    tic = time.time

    for phase in ['test'] if args.test else ['train','test']:
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # Iterate over data.
        num_iters = 0
        for lr,hr in loaders[phase]:
            lr = lr.to(device)
            hr = hr.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                pred = model(lr)
                vars = {'pred':pred, 'lr':lr, 'hr':hr}
                loss = Loss(pred,hr)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # compute statistics
            psnr_epoch += PSNR(pred,hr)
            loss_epoch += loss

        # logging metrics
        psnr_epoch /= num_iters
        loss_epoch /= num_iters
        tac = time.time()

        # tensorboard
        if args.tensorboard:
            writer.add_scalar(f'Loss/{loss_epoch}', loss_epoch, epoch)
            writer.add_scalar(f'Psnr/{psnr_epoch}', psnr_epoch, epoch)

        # console logging
        epoch_log = f'{phase} {epoch} | {(tac-tic):0.2f}s ({(tac-tic)/num_iters:0.2f}s/iter) | psnr {psnr_epoch:0.3f} |'
        if device!=torch.device('cpu'):
            epoch_log += f' | gpu {show_gpu_mem()[1]:0.1f} Mb'
        tqdm.write(epoch_log)

        if phase == 'test':
            if psnr_epoch > best_psnr:
                best_psnr = psnr_epoch
                is_best = True
            else:
                is_best = False

    # deep copy the model
    save_checkpoint(model, optimizer, args, epoch, psnr_epoch, is_best, ckpt_path)

if args.tensorboard:
    writer.close()