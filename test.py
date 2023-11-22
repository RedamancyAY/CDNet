

import os

from datetime import datetime
import argparse
from pathlib import Path

from cv2 import stereoCalibrate
from torchsummary.torchsummary import summary


from toolkits.data import FrameFaceDataset
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist


from toolkits import utils, split
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

from config import config
import time
from model import xception


def distance(v1, v2):
    tmp = v1-v2
    tmp = torch.dot(tmp, tmp)
    tmp = torch.sum(tmp)
    return tmp


if __name__ == '__main__':
    v1 = torch.Tensor([1, 2, 3, 4, 5])
    v2 = torch.zeros(5)
    d = distance(v1, v2)
    print(d)
