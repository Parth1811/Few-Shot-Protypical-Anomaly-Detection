import argparse
import copy
import os
import warnings
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.serialization import SourceChangeWarning

from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE
from model.utils import SceneLoader
from utils import *


# Muting the source change warnings
warnings.filterwarnings("ignore", category=SourceChangeWarning)
check_running(__file__)

# Parsing the input command arguments
parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--inner_lr', type=float, default=2e-4, help='Initial learning rate for inner update')
parser.add_argument('--outer_lr', type=float, default=2e-5, help='Initial learning rate for outer update')
parser.add_argument('--time_step', type=int, default=4, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='shanghaitech', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--log_dir', type=str, default='log', help='directory of log')
parser.add_argument('--model_dir', type=str, default=None, help='directory of model')
parser.add_argument('--m_items_dir', type=str, default=None, help='directory of model')
parser.add_argument('--k_shots', type=int, default=4, help='Number of K shots allowed in few shot learning')
parser.add_argument('--N', type=int, default=4, help='Number of Scenes sampled at a time')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for the training loop')
parser.add_argument('--seperate_save_files_per_epochs', type=bool, default=False, help='Flag which determines wether or not to overide model files while saving')
parser.add_argument('--single_scene_database', type=bool, default=None, help='Flag for single scene data base')
parser.add_argument('--descripton', type=str, default=None, help='Any special description for this run')
args = parser.parse_args()

# Setting up the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]


# Setting up pytorch environment
torch.manual_seed(2021)
torch.backends.cudnn.enabled = True     # make sure to use cudnn for computational performance


# Loading dataset
if args.single_scene_database is None and (args.dataset_type == "ped2" or args.dataset_type == "avenue"):
    args.single_scene_database = True
else:
    args.single_scene_database = False
train_folder = os.path.join(args.dataset_path, args.dataset_type, "training/frames")

train_dataset = SceneLoader(
    train_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h,
    resize_width=args.w,
    k_shots=args.k_shots,
    time_step=args.time_step,
    num_workers=args.num_workers,
    single_scene=args.single_scene_database
)


# Setting up the model, memory items and optimizers
if args.model_dir is not None:
    model = torch.load(args.model_dir)
else:
    model = convAE(args.c, args.time_step + 1, args.msize, args.fdim, args.mdim)
model = model.cuda()
model.train()

if args.m_items_dir is not None:
    m_items = torch.load(args.m_items_dir)
else:
    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1)
m_items = m_items.cuda()

params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.outer_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
dummy_inner_optimizer = torch.optim.Adam(params, lr=args.inner_lr)
inner_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(dummy_inner_optimizer, args.epochs//5 + 1)
loss_func_mse = nn.MSELoss(reduction='none')


# attaching date and time to make names unique
run_start_time = datetime.now()
def attach_datetime(string):
    return string + run_start_time.strftime(f"_%d-%m-%H-%M")


# Setting up the logging modules
# Log levels
# Inside Iteration 15
# Epoch End 35
# Model save 45
# Log Save 25
progressbar = tqdm(range(args.epochs * args.iterations), desc="Training "+args.dataset_type,
                   ascii=False, dynamic_ncols=True, colour="yellow")
log_dir = os.path.join(args.log_dir, args.dataset_type)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, attach_datetime('run') + '.log')
logger = setup_logger(log_file_path)


# Training
logger.info(args)
logger.info('Training has Started')
for epoch in range(args.epochs):

    # Repeating iteration for Meta-Training
    for iter in range(args.iterations):
        if iter % (args.iterations // 10) == 0:
            logger.log(15, "Epoch: %d : Iteration: %d" % (epoch, iter))

        # Sampling N scenes from the dataset
        try:
            scenes = train_dataset.get_dataloaders_of_N_random_scenes(args.N)
        except ValueError:
            train_dataset.reset_iters()
            logger.log(15, "Epoch: %d : Iteration: %d : Recreated the SceneLoader object ", epoch, iter)
            scenes = train_dataset.get_dataloaders_of_N_random_scenes(args.N)

        # Clearing gradients from previous pass
        optimizer.zero_grad()

        # Inner update loop for each scene
        for scene, train_batch in scenes:

            # Cloning the model, which would be used in inner update
            inner_model = copy.deepcopy(model)
            inner_params_encoder = list(inner_model.encoder.parameters())
            inner_params_decoder = list(inner_model.decoder.parameters())
            inner_params = inner_params_encoder + inner_params_decoder
            inner_optimizer = torch.optim.Adam(inner_params, lr=dummy_inner_optimizer.param_groups[0]['lr'])

            # Sampling k samples for training and validation each
            try:
                imgs = Variable(next(train_batch)).cuda()
                imgs_val = Variable(next(train_batch)).cuda()
            except StopIteration as e:
                if scene in train_dataset.scenes:
                    train_dataset.scenes.remove(scene)
                    logger.log(15, "Epoch: %d : Iteration: %d : Removed scene '%s' from the list of Scenes", epoch, iter, scene)
                continue
            except Exception as e:
                logger.error(str(e))
                continue

            # Forward pass
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = inner_model.forward(imgs[:, 0:3 * args.time_step], m_items, True)

            # Performing the inner update on clone model
            inner_optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 3 * args.time_step:]))
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            loss.backward(retain_graph=True)
            inner_optimizer.step()

            # Computing gradient of updated model on the validation set
            inner_optimizer.zero_grad()
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = inner_model.forward(imgs_val[:, 0:3 * args.time_step], m_items, True)
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs_val[:, 3 * args.time_step:]))
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            loss.backward(retain_graph=True)

            # Accumulation the gradients from each scene to perform a outer update
            for i in range(len(params)):
                if params[i].grad is None:
                    params[i].grad = copy.deepcopy(inner_params[i].grad)
                else:
                    params[i].grad += inner_params[i].grad

        # Perfoming outer update
        optimizer.step()
        inner_scheduler.step()
        progressbar.update(1)

    scheduler.step()

    print('----------------------------------------')
    logger.log(35, 'Epoch: ' + str(epoch + 1))
    logger.log(35, 'Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')
    if args.seperate_save_files_per_epochs:
        torch.save(model, os.path.join(log_dir, 'model_%d.pth' % epoch))
        logger.log(45, "Saved Model in " + os.path.join(log_dir, 'model_%d.pth' % epoch))
        torch.save(m_items, os.path.join(log_dir, 'keys_%d.pt' % epoch))
        logger.log(45, "Saved Memory Items in " + os.path.join(log_dir, 'keys_%d.pt' % epoch))
    else:
        torch.save(model, os.path.join(log_dir, 'model.pth'))
        logger.log(45, "Saved Model in " + os.path.join(log_dir, 'model.pth'))
        torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
        logger.log(45, "Saved Memory Items in " + os.path.join(log_dir, 'keys.pt'))

logger.info('Training is finished')
logger.log(25, "Saved log file " + log_file_path)
progressbar.close()
