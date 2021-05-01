import copy
import logging
import math
import os
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):

    return 10 * math.log10(1 / mse)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)

    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(- error))
    score = (torch.sum(normal * loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list_result


class ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy.copy(record)
        levelno = myrecord.levelno
        if(levelno >= 50):  # CRITICAL / FATAL
            color = '\x1b[31m'  # red
        elif(levelno >= 40):  # ERROR
            color = '\x1b[31m'  # red
        elif(levelno >= 30):  # WARNING
            color = '\x1b[33m'  # yellow
        elif(levelno >= 20):  # INFO
            color = '\x1b[36m'  # cyan
        elif(levelno >= 15):  # DEBUG
            color = '\x1b[35m'  # pink
        elif(levelno == 10):  # notification
            color = '\x1b[32m'  # green
        else:  # NOTSET and anything else
            color = '\x1b[0m'  # normal
        myrecord.msg = color + str(myrecord.msg) + '\x1b[0m'  # normal
        logging.StreamHandler.emit(self, myrecord)


def setup_logger(log_file_path):
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(message)s',
        datefmt='%m/%d/%y %I:%M:%S %p'
    )

    file_handler = logging.FileHandler(
        log_file_path,
        mode='w'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)
    out_stream = type("TqdmStream", (), {'file': sys.stdout, 'write':write})()

    stdout_handler = ColoredConsoleHandler(out_stream)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    logging.addLevelName(15, "Inside Iteration")     # pink
    logging.addLevelName(35, "Epoch End")            # yellow
    logging.addLevelName(45, "Model Save")           # red
    logging.addLevelName(25, "Log saved")            # cyan
    logging.addLevelName(12, "Notification")            # cyan

    return logger

def check_running(file):
    n = 0
    with os.popen('ps aux | grep "python %s" | wc -l' % file) as f:
        n = int(f.readlines()[0].split('\n')[0])

    if n > 4:
        res = input("There might be already running session.\n    \
View output of ongoing session [y]\n    Run new session anyways [n] \n")
        if res == "y":
            ret = os.system("tail -f nohup.out")
            sys.exit()
        elif res == "n":
            pass
        else:
            sys.exit()
