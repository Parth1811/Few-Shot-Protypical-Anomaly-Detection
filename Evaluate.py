import argparse
import copy
import os
import warnings
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.serialization import SourceChangeWarning

from model.utils import SceneLoader
from utils import (AUC, anomaly_score_list, anomaly_score_list_inv,
                   check_running, point_score, psnr, score_sum, setup_logger)

# Muting the source change warnings
warnings.filterwarnings("ignore", category=SourceChangeWarning)
check_running(__file__)

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for update')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--time_step', type=int, default=4, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--log_dir', type=str, default='log', help='directory of log')
parser.add_argument('--dataset_type', type=str, default='shanghaitech', help='type of dataset: ped2, avenue, shanghaitech')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--k_shots', type=int, default=4, help='Number of K shots allowed in few shot learning')
parser.add_argument('--N', type=int, default=4, help='Number of Scenes sampled at a time')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for the training loop')
parser.add_argument('--single_scene_database', type=bool, default=None, help='Flag changes the behaviour of Dataloader to load when there is only one scene and no seperate scene folders')
parser.add_argument('--save_anomaly_list', type=bool, default=True, help='Flag anomaly list of individual videos')
parser.add_argument('--descripton', type=str, default=None, help='Any special description for this run')
args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True     # make sure to use cudnn for computational performance


# Loading dataset
if args.single_scene_database is None and (args.dataset_type == "ped2" or args.dataset_type == "avenue"):
    args.single_scene_database = True
else:
    args.single_scene_database = False
test_folder = os.path.join(args.dataset_path, args.dataset_type, "testing/frames/")
test_batch = SceneLoader(
    test_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h,
    resize_width=args.w,
    k_shots=1,
    time_step=args.time_step,
    num_workers=args.num_workers,
    single_scene=args.single_scene_database,
    shuffle=False,
    drop_last=False
)

# loading labels
labels = np.load('./data/frame_labels_%s.npy' % args.dataset_type)[0]
if len(labels) != len(test_batch) + (test_batch.get_video_count() * args.time_step):
    raise ValueError("The length of dataset doesn't match the original length for which the labels are avaibale.")
label_list, video_ref_dict = test_batch.process_label_list(labels)

# attaching date and time to make names unique
run_start_time = datetime.now()
def attach_datetime(string):
    return string + run_start_time.strftime(f"%d-%m-%H-%M")


# Setting up the logging modules
# Log levels
# Inside Iteration 15
# Epoch End 35
# Model save 45
# Log Save 25
progressbar = tqdm(range(len(test_batch)), desc="Evaluating "+args.dataset_type,
                   ascii=False, dynamic_ncols=True, colour="blue")
log_dir = os.path.join(args.log_dir, args.dataset_type)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, attach_datetime('test_') + '.log')
logger = setup_logger(log_file_path)

# Loading the trained model
model = torch.load(args.model_dir)
model = model.cuda()
model.train()

m_items = torch.load(args.m_items_dir)
m_items = m_items.cuda()

params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.lr)
loss_func_mse = nn.MSELoss(reduction='none')
psnr_list = {}
feature_distance_list = {}
curr_video_name = 'default'
prev_k = 0
k = -1
anomaly_score_total_list = []

logger.info(args)
logger.info('Evaluation has started')
for s_id, scene in enumerate(sorted(test_batch.scenes), 1):
    logger.log(15, "Starting Evalution for Scene:%s; %d of %d" %
               (scene, s_id, len(test_batch.scenes)))
    imgs = []
    for _ in range(args.k_shots):
        imgs.append(next(test_batch.dataloader_iters[scene][1]))
    imgs = np.concatenate(imgs, axis=0)
    imgs = torch.from_numpy(imgs)
    imgs = Variable(imgs).cuda()
    test_batch.reset_iters()

    outputs, _, _, _, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
        imgs[:, 0:3 * args.time_step], m_items, True)

    optimizer.zero_grad()
    loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 3 * args.time_step:]))
    loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
    loss.backward(retain_graph=True)
    optimizer.step()

    inner_model = copy.deepcopy(model)
    inner_model.eval()
    inner_model.cuda()

    scene_log_dir = os.path.join(log_dir, attach_datetime(""), scene)
    os.makedirs(os.path.join(log_dir, attach_datetime(""), scene))
    video_num = 0

    for k, (imgs) in enumerate(test_batch.dataloader_iters[scene][1], k + 1):

        if k in video_ref_dict:
            prev_k = k
            curr_video_name = video_ref_dict[k]
            psnr_list[curr_video_name] = []
            feature_distance_list[curr_video_name] = []
            video_num += 1

        imgs = Variable(imgs).cuda()

        outputs, feas, updated_feas, m_items, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = inner_model.forward(imgs[:, 0:3 * args.time_step], m_items, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * args.time_step:] + 1) / 2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:, 3 * args.time_step:])

        if point_sc < args.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 1)      # b X h X w X d
            m_items = model.memory.update(query, m_items, False)

        psnr_list[curr_video_name].append(psnr(mse_imgs))
        feature_distance_list[curr_video_name].append(mse_feas)

        if (k + 1) % len(test_batch) in video_ref_dict:
            anomaly_score_list_for_video = score_sum(anomaly_score_list(
                psnr_list[curr_video_name]), anomaly_score_list_inv(feature_distance_list[curr_video_name]), args.alpha)
            if args.save_anomaly_list:
                np.save(os.path.join(scene_log_dir, "%s.npy" %
                        curr_video_name), [anomaly_score_list_for_video, label_list[prev_k:k + 1]])
                logger.log(25, "Score list at %s %s.npy; %d of %d" % (
                    os.path.basename(os.path.normpath(args.log_dir)), curr_video_name, video_num, len(test_batch.scenes_dataloader[scene].dataset.videos)))
            else:
                logger.log(12, "Processed Scene: %s, Video: %s; %d of %d" % (
                    scene, curr_video_name, video_num, len(test_batch.scenes_dataloader[scene].dataset.videos)))
            anomaly_score_total_list += anomaly_score_list_for_video

        progressbar.update(1)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)
accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - label_list, 0))

logger.log(35, 'AUC: ' + str(accuracy * 100) + '%')
logger.info('Evaluation is finished')
logger.log(25, "Saved log file " + log_file_path)
progressbar.close()
