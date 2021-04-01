import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import SceneLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='shanghaitech', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--model_dir', type=str, default=None, help='directory of model')
parser.add_argument('--m_items_dir', type=str, default=None, help='directory of model')
parser.add_argument('--k_shots', type=int, default=4, help='Number of K shots allowed in few shot learning')

args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path,args.dataset_type,"training/scenes")

# Loading dataset
train_dataset = SceneLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, k_shots=args.k_shots,time_step=args.t_length-1)


# Model setting
if args.model_dir is not None:
    model = torch.load(args.model_dir, map_location=torch.device('cpu'))
else:
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
params_encoder =  list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model#.cuda()


# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
# f = open(os.path.join(log_dir, 'log.txt'),'w')
# sys.stdout= f

loss_func_mse = nn.MSELoss(reduction='none')
if args.m_items_dir is not None:
    m_items = torch.load(args.m_items_dir, map_location=torch.device('cpu'))
else:
    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1)#.cuda() # Initialize the memory items

# Training

iterations = 1000
N = 4

for epoch in range(args.epochs):
    labels_list = []
    model.train()
    start = time.time()


    for iter in range(iterations):

        scenes = train_dataset.get_dataloaders_of_N_random_scenes(N)
        optimizer.zero_grad()

        for train_batch, val_batch in scenes:

            inner_model = copy.deepcopy(model)
            inner_params_encoder =  list(inner_model.encoder.parameters())
            inner_params_decoder = list(inner_model.decoder.parameters())
            inner_params = inner_params_encoder + inner_params_decoder
            inner_optimizer = torch.optim.Adam(inner_params, lr = args.lr)

            try:
                imgs = Variable(next(train_batch))#.cuda()
                imgs_val = Variable(next(val_batch))#.cuda()
            except StopIteration:
                continue

            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = inner_model.forward(imgs[:,0:12], m_items, True)

            inner_optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            loss.backward(retain_graph=True)
            inner_optimizer.step()

            inner_optimizer.zero_grad()

            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = inner_model.forward(imgs_val[:,0:12], m_items, True)

            loss_pixel = torch.mean(loss_func_mse(outputs, imgs_val[:,12:]))
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            loss.backward(retain_graph=True)
            for i in range(len(params)):
                if params[i].grad is None:
                    params[i].grad = copy.deepcopy(inner_params[i].grad)
                else:
                    params[i].grad += inner_params[i].grad

        optimizer.step()


    scheduler.step()

    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')

print('Training is finished')
# Save the model and the memory items
torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))

sys.stdout = orig_stdout
f.close()