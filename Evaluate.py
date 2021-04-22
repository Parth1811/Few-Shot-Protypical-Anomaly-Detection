import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.utils import SceneLoader
from utils import (AUC, anomaly_score_list, anomaly_score_list_inv,
                   point_score, psnr, score_sum)

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
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
parser.add_argument('--dataset_type', type=str, default='avenue', help='type of dataset: ped2, avenue, shanghaitech')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--k_shots', type=int, default=4, help='Number of K shots allowed in few shot learning')
parser.add_argument('--N', type=int, default=4, help='Number of Scenes sampled at a time')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for the training loop')
parser.add_argument('--single_scene_database', type=bool, default=None, help='Flag changes the behaviour of Dataloader to load when there is only one scene and no seperate scene folders')

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

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
model.train()

m_items = torch.load(args.m_items_dir)
m_items.cuda()

params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.outer_lr)
loss_func_mse = nn.MSELoss(reduction='none')
psnr_list = {}
feature_distance_list = {}
curr_video_name = 'default'
k = 0


for scene in test_batch.scenes:
    imgs = []
    for _ in range(args.k_shots):
        imgs.append(next(test_batch.dataloader_iters[scene][1]))
    imgs = np.concatenate(imgs, axis=0)
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

    return inner_model


    for k, (imgs) in enumerate(test_batch.dataloader_iters[scene][1], k):

        if k in video_ref_dict:
            curr_video_name = video_ref_dict[k]

        imgs = Variable(imgs)
        print("doing something ", k)

        outputs, feas, updated_feas, m_items, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = innner_model.forward(imgs[:, 0:3 * args.time_step], m_items, False)
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


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video_name in sorted(video_ref_dict.values()):
    anomaly_score_total_list += score_sum(anomaly_score_list(
        psnr_list[video_name]), anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - label_list, 0))

print('The result of ', args.dataset_type)
print('AUC: ', accuracy * 100, '%')
