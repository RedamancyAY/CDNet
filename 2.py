'''
deepfake detection multicard 

@author hzm
@date 2021.12.1
'''

import os

from datetime import datetime
import argparse
from pathlib import Path
from statistics import median

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
import torch.nn.functional as F

from config import config
import time
from model import xception
from model import models1

###################
# 分布式训练
###################

# ! 流程标识
# * 方法具体执行解释
# ? 测试代码注释


'''
:step 1     进入分布式训练，不同的进程先展示自己所属的id=对应所用的GPU
:step 2     进行初始参数配置  
:step 3     设置优化器和学习率调节器
:step 4     模型加载
:step 5     加载图片载入要用的transform
:step 6     数据集处理
:step 7     初始化tensorboard
:step 8     开始训练
'''
# Todo
'''
1.训练的模型再次载入报错                                           fix： 2021.12.2  15:05
2.再次载入后，在优化器的step阶段，变量不在一个tensor上，有些在cpu上   fix：2021.12.2  15:05
3.config类的编写                                                    done: 2021.12.2     22:00
4.保存模型的载入有问题，其实是ddp的保存方式导致最终的保存模型在dict上名字对不上     fix：2021.12.2 19:40
5.调整学习率scheduler
6.设计新的蒸馏方案
7.结合蒸馏得到feature合成最终的合体模型
8.当然啦，老师可以是多个
'''


def dist_train(gpu, args):

    torch.autograd.set_detect_anomaly(True)
    # !step 1
    rank = gpu  # 当前进程号
    print('Rank id: ', rank)

    # !step 2
    # *将args获取的参数转化为变量

    train_datasets = args.traindb
    train_datasets2 = args.traindb2
    trainIndex = args.trainIndex
    mode = args.mode
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir
    dfdc_df_path = args.dfdc_faces_df_path
    dfdc_faces_dir = args.dfdc_faces_dir
    celebdf_df_path = args.celebdf_faces_df_path
    celebdf_faces_dir = args.celebdf_faces_dir
    face_policy = args.face
    box_size = args.size
    batch_size = args.batch
    initial_lr = args.lr
    validation_interval = args.valint
    patience = args.patience

    # ?max_train_samples = args.trainsamples
    log_interval = args.logint
    num_workers = args.workers
    seed = args.seed
    debug = args.debug

    # ?enable_attention = args.attention
    weights_folder = args.models_dir
    logs_folder = args.log_dir
    world_size = args.world_size
    backend = args.backend
    init_method = args.init_method
    epoch_run = args.epochs
    model_period = args.modelperiod
    tagnote = args.tagnote

    initial_model = args.index
    # suffix = args.suffix

    # *初始化进程组，决定进程的通信方式，自己的进程标志
    dist.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    torch.manual_seed(0)

    # *获取nodel_class,生成model

    model_t_class = getattr(xception, args.net_t)

    model_t = model_t_class()
    transformer = utils.get_transformer(face_policy=face_policy, patch_size=299,
                                        net_normalizer=model_t.get_normalizer(), train=True)

    # *生成model对应的tag
    tag = utils.make_train_tag(net_class=model_t_class,
                               traindb=train_datasets,
                               face_policy=face_policy,
                               patch_size=299,
                               cuttings_size=32,
                               seed=seed,
                               debug=debug,
                               note=tagnote
                               )

    # *生成saved model的路径还有,生成文件夹
    bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, tag, 'last.pth')
    periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')
    path_list = [bestval_path, last_path, periodic_path.format(initial_model)]
    os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

    # !step3

    optimizer = torch.optim.Adam(
        model_t.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=initial_lr*1e-7,
    )

    # !step4
    # *模型超参配置
    val_loss = min_val_loss = 10

    epoch = iteration = 0
    model_state = None
    opt_state = None

    # *对模型进行加载
    # TODO 编写模型加载模块

    epoch, iteration = load_model(model_t, optimizer, path_list, mode,
                                  initial_model)

    print(epoch)

    model_t = model_t.cuda(gpu)

    model_t = nn.parallel.DistributedDataParallel(
        model_t, device_ids=[gpu], find_unused_parameters=True)

    # 将所有optimizer的数据放回到cuda上
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(gpu)

    # !step5
    # *生成数据增强用到的transform

    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # *同步batch归一化
    if args.syncbn:
        model_t = nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
        if gpu == 0:
            print('Use SyncBN in training')
    torch.cuda.set_device(gpu)

    # !step6

    # *加载数据
    print("Loading data")

    # *从总dfs中，提取所需的dfs
    # dfs_train, dfs_val = split.make_splits_FFPP(ffpp_df_path, train_datasets)

    dfs_train, dfs_val = split.make_split_FFPP(
        ffpp_df_path, train_datasets)
    # dfs_train1, dfs_val1 = split.make_splits_celebdf(
    #     celebdf_df_path, train_datasets2)

    # dfs_train_2, dfs_val_2 = split.make_splits_celebdf(celebdf_df_path,)

    # *制作iterable数据集，DDP不能用iterable的数据集，数据集应实现方法__getitem__
    train_dataset = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_train,
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=box_size,
                                     )
    val_dataset_1 = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_val,
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=box_size,
                                     )
    # val_dataset_2 = FrameFaceDataset(root=celebdf_faces_dir,
    #                                  df=dfs_val1,
    #                                  scale=face_policy,
    #                                  transformer=transformer,
    #                                  size=box_size,
    #                                  )
    # val_dataset_2 = FrameFaceDataset(root=celebdf_faces_dir,
    #                                  df=dfs_)

    # 验证生成的数据集长度
    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    if len(val_dataset_1) == 0:
        print('No validation samples. Halt.')
        return

    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation samples: {}'.format(len(val_dataset_1)))

    # *将数据集提供给sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_1,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    # val_sampler1 = torch.utils.data.distributed.DistributedSampler(val_dataset_2,
    #                                                                num_replicas=world_size,
    #                                                                rank=rank)

    # *生成Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset_1, num_workers=num_workers, batch_size=batch_size, pin_memory=True,
        sampler=val_sampler)

    # val_loader1 = torch.utils.data.DataLoader(
    #     val_dataset_2, num_workers=num_workers, batch_size=batch_size, pin_memory=True,
    #     sampler=val_sampler1)

    # !step 7
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # *若是一开始训练，log已经存在，将其删掉
        shutil.rmtree(logdir, ignore_errors=True)
    tb = SummaryWriter(logdir=logdir)

    # !step 8 初始化原始center

    while epoch != epoch_run:
        # ?optimizer.zero_grad()

        train_loss = train_num = 0
        train_pred_list = []
        train_labels_list = []
        current = time.time()
        train_batch_loss = 0
        acc_total1 = 0
        acc_total2 = 0
        for train_batch in tqdm(train_loader, desc='Epoch {:03d} '.format(epoch), leave=False,
                                total=len(train_loader)):

            model_t.train()
            (batch_data, batch_labels), batch_df = train_batch
            train_batch_num = len(batch_labels)
            # *param train_num 用于统计训练总数
            train_num += train_batch_num

            start = time.time()
            # print(start-current)

            train_batch_loss, pred = batch_forward(
                model_t, batch_data, batch_labels, batch_df)

            # _, seq = torch.min(train_batch_pred, 1)
            # seq = train_batch_pred >= 0.5

            train_labels_list.append(batch_labels.cpu())
            train_pred_list.append(pred.cpu())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num
            # acc_total1 = acc_total1+acc1*train_batch_num

            # *运用优化器
            train_batch_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            # # *当达到周期时，选择震荡学习率，从而学习更优模型
            # if iteration > 10000 and (iteration % vib_period) == 0:
            #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * \
            #         np.random.randint(2, vib_factor)

            # *记录训练阶段数据，保存模型
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                # acc_total1 /= train_num
                # acc_total2 /= train_num
                train_labels = np.concatenate(train_labels_list).flatten()
                train_pred = np.concatenate(train_pred_list).flatten()

                train_acc = accuracy_score(train_labels, train_pred)
                train_roc_auc = roc_auc_score(train_labels, train_pred)

                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)
                tb.add_scalar('train/acc', train_acc, iteration)
                # tb.add_scalar('train/acc2', acc_total2, iteration)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.flush()

                # *500个batch，会保存一次model
                if (iteration % model_period == 0):

                    # save_model_v2(model_t, optimizer, train_loss, val_loss,
                    #               iteration, batch_size, epoch, periodic_path.format(iteration))
                    save_model_v2(model_t, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, last_path)

                acc_total1 = acc_total2 = train_loss = train_num = 0

                train_labels_list = []
                train_pred_list = []

            # *对模型进行验证
            if iteration > 0 and (iteration % validation_interval == 0):

                # train_labels = np.concatenate(train_labels_list)
                # train_pred = np.concatenate(train_pred_list)
                # train_labels_list = []
                # train_pred_list = []

                # train_roc_auc = roc_auc_score(train_labels, train_pred)
                # train_f1 = f1_score(train_labels, train_pred)
                # tb.add_scalar('train/f1', train_f1, iteration)
                # tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                # tb.add_pr_curve('train/pr', train_labels,
                #                 train_pred, iteration)

                # *Validation

                val_loss = validation_routine(
                    model_t, val_loader, 'val_FFPP', iteration, tb)
                # val_loss1 = validation_routine(
                #     model_t, val_loader1, 'val_celeb', iteration, tb)
                tb.flush()

                # *根据loss调整
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model_v2(model_t, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, bestval_path)
            # *每迭代一个batch +1
            iteration = iteration + 1
            # current = time.time()
        # val_acc1, val_loss1 = validation_v1(
        #     model_t, val_loader, iteration, tb, tag='FFPP_')
        # val_acc2, val_loss2 = validation_v1(model_t, val_loader1,
        #                                     iteration, tb, tag='celebdf_')
        # val_loss = 0.5*(val_loss1+val_loss2)
        # tb.add_scalar('val/loss', val_loss, iteration)
        # lr_scheduler.step(val_loss)
        # if val_loss < min_val_loss:
        #     min_val_loss = val_loss

        #     save_model_v2(model_t, optimizer, train_loss, val_loss,
        #                   iteration, batch_size, epoch, bestval_path)
        epoch = epoch + 1


'''
method definition:
------------------
:param net          训练所用模型
:param device       训练所用的cuda编号
:param criterion    训练所用的损失函数
:param data         训练用的batch大小图片，ndarray
:param labels       训练用的标签
-----------------
:return loss[float]          训练所产生的损失    
:return pred[ndarray]        训练的预测值
'''


def batch_forward(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, paths: list[Path]):

    # correct,total = 0,0
    data = data.cuda(non_blocking=True)

    labels = labels.cuda(non_blocking=True)

    # labels_map = labels_mapGen(labels)
    # labels_map, seq = data2seq(data, labels)

    fea = model(data)
    # fea = torch.softmax(fea, dim=1)
    # print(pred[5])
    # print(labels_map[5])

    # pred = nn.Sigmoid()(pred)

    # _, predicted = torch.max(pred, 1)
    # correct += (predicted == labels).sum().item()
    # total = len(labels)
    # print(correct/total)
    # labels = labels.long().squeeze()
    # loss = nn.CrossEntropyLoss()(pred, labels)

    # pred = torch.sigmoid(pred)

    # t_out = torch.sigmoid(t_out)
    # # # 将网络的输出转化为[0,1]，同时转为nadarray
    # pred = pred.detach().cpu().numpy()
    # t_out = t_out.detach().cpu().numpy()
    # # 计算Loss
    # pred = pred.detach().cpu().numpy()
    # train_predL = [int(item > 0.2) for item in pred]
    # for i in range(len(train_predL)):
    #     if(labels[i] != train_predL[i] ):
    #         # print(paths[i])
    # print(labels.long().flatten())
    # print(fea)
    # 先用CE，后面再进行改进
    loss = nn.BCEWithLogitsLoss()(fea, labels.float())
    # loss = nn.BCEWithLogitsLoss()(fea, labels)
    # fea = nn.Softmax(fea, dim=1)
    pred = torch.where(fea[:, 0] > 0.5, 1, 0)
    # acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
    # auc = roc_auc_score(labels.cpu().numpy(), fea[:, 1].cpu().detach().numpy())
    # pred = nn.Sigmoid()(pred)

    # pred = pred.detach().cpu().numpy()

    # loss = Loss_cal(pred, t_out, w, labels, iteration, tb, flag)

    return loss, pred


def batch_forward_val(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, paths: list[Path], lr, iteration):
    data = data.float().cuda(non_blocking=True)

    labels = labels.cuda(non_blocking=True)

    labels_map = labels_mapGen(labels)

    fea, centers = model(data, labels_map, lr)

    acc_seperate = cluster_acc(fea, centers, labels_map)

    acc_togrther = cluster_acc_v1(fea, centers, labels_map)

    loss_val = clusterloss_val(fea, labels_map, centers, lr, iteration)

    return acc_seperate, acc_togrther, loss_val


def cluster_acc_v1(fea, centers, labels_map):
    labels_map = labels_map.reshape(-1, 4)
    labels_map = labels_map[:, 0]
    centerT, centerF = centers

    fea2centerT = centerT - fea
    fea2centerF = centerF - fea

    fea2centerT = torch.mul(fea2centerT, fea2centerT)
    fea2centerF = torch.mul(fea2centerF, fea2centerF)

    fea2centerT = torch.sum(fea2centerT, 1, keepdim=True)
    fea2centerF = torch.sum(fea2centerF, 1, keepdim=True)

    distance = torch.cat((fea2centerT, fea2centerF), 1)
    distance = distance.reshape(-1, 4, 2)
    distance = torch.sum(distance, dim=1)
    # distance =

    predict = torch.max(distance, 1)[1]

    acc = accuracy_score(predict.cpu(), labels_map.cpu())

    return acc


def cluster_acc(fea, centers, labels_map):
    centerT, centerF = centers

    fea2centerT = centerT - fea
    fea2centerF = centerF - fea

    fea2centerT = torch.mul(fea2centerT, fea2centerT)
    fea2centerF = torch.mul(fea2centerF, fea2centerF)

    fea2centerT = torch.sum(fea2centerT, 1, keepdim=True)
    fea2centerF = torch.sum(fea2centerF, 1, keepdim=True)
    max_d1 = torch.max(fea2centerT)
    max_d2 = torch.max(fea2centerF)

    distance = torch.cat((fea2centerT, fea2centerF), 1)

    predict = torch.max(distance, 1)[1]

    acc = accuracy_score(predict.cpu(), labels_map.cpu())
    return acc


def labels_mapGen(labels_map: torch.tensor):
    tmp = labels_map.repeat(1, 4)
    tmp = tmp.view(-1)
    return tmp


# def center_update(centers, fea, labels_map, iteration, lr):
#     lr_center = 100*lr
#     centerT, centerF = centers

#     tmp1 = fea[torch.where(labels_map > 0)[0]]
#     tmp2 = fea[torch.where(labels_map == 0)[0]]

#     centerT_currentB = torch.mean(tmp1, 0)
#     centerF_currentB = torch.mean(tmp2, 0)

#     if iteration == 0:
#         centerT = centerT_currentB
#         centerF = centerF_currentB
#     else:
#         centerT = (1-lr_center)*centerT+lr_center*centerT_currentB
#         centerF = (1-lr_center)*centerF+lr_center*centerF_currentB

#     return centerT, centerF


def clusterloss(fea: torch.Tensor, labels_map: torch.Tensor, centers, lr, iteration):

    # tmp1 = F.normalize(seq1, 2, 1)
    # tmp1 = tmp1.unsqueeze(1)
    # tmp2 = F.normalize(seq2, 2, 1)
    # tmp2 = tmp2.unsqueeze(1)
    # tmp2 = tmp2.transpose(1, 2)
    # sim = torch.bmm(tmp1, tmp2)
    # sim = (sim+1)/2
    # sim = 1-sim
    # # loss = torch.mul(sim, sim)
    # loss = torch.sum(sim)
    loss = 0

    centerT, centerF = centers

    loss_dif = 0
    loss_same = 0

    # for i in range(16):
    #     for j in range(16):
    #         d = distance(centerT, fea[i][j])
    #         maxd_1 = max(maxd_1, d)
    #         loss_same += d
    #     for j in range(16, 32):
    #         d = distance(centerF, fea[i][j])
    #         maxd_2 = max(maxd_2, d)
    #         loss_same += d

    # for i in range(16, 24):
    #     for j in range(32):
    #         d = distance(centerT, fea[i][j])
    #         maxd_1 = max(maxd_1, d)
    #         loss_same += d

    # for i in range(24, 32):
    #     for j in range(32):
    #         d = distance(centerF, fea[i][j])
    #         maxd_2 = max(maxd_2, d)
    #         loss_same += d

    tmp1 = fea[torch.where(labels_map > 0)]
    tmp2 = fea[torch.where(labels_map == 0)]

    tmp1 = centerT-tmp1
    tmp1 = torch.mul(tmp1, tmp1)
    tmp1 = torch.sum(tmp1, dim=1)

    median_1 = torch.median(tmp1)
    median_1 = torch.sqrt(median_1)

    loss_same = loss_same+torch.sum(tmp1)

    tmp2 = centerF-tmp2
    tmp2 = torch.mul(tmp2, tmp2)
    tmp2 = torch.sum(tmp2, dim=1)
    median_2 = torch.median(tmp2)
    median_2 = torch.sqrt(median_2)

    loss_same = loss_same+torch.sum(tmp2)
    loss_same = loss_same/512
    median_1 = median_1.detach()
    median_2 = median_2.detach()

    # ! 待跑
    if iteration > 9000:

        loss_dif = 1/(distance(centerT, centerF))
        if (distance(centerT, centerF) > (loss_same*10)):
            loss_dif = 0

    else:
        loss_dif = 1/(distance(centerT, centerF))

    # if iteration < 1800:
    #     loss = loss_dif
    # else:
    #     while loss_same > loss_dif:
    #         loss_same = loss_same/2
    #     loss = loss_same+loss_dif

    # while loss_same > loss_dif:
    #     loss_same = loss_same/2
    loss = loss_same+loss_dif

    if iteration % 100 == 0:

        print("distance = %f" % distance(centerT, centerF))
        print("loss_dif = %f" % loss_dif)
        print("loss_same = %f" % loss_same)

    return loss


def clusterloss_val(fea: torch.Tensor, labels_map: torch.Tensor, centers, lr, iteration):
    loss = 0

    centerT, centerF = centers

    loss_same = 0

    # for i in range(16):
    #     for j in range(16):
    #         d = distance(centerT, fea[i][j])
    #         maxd_1 = max(maxd_1, d)
    #         loss_same += d
    #     for j in range(16, 32):
    #         d = distance(centerF, fea[i][j])
    #         maxd_2 = max(maxd_2, d)
    #         loss_same += d

    # for i in range(16, 24):
    #     for j in range(32):
    #         d = distance(centerT, fea[i][j])
    #         maxd_1 = max(maxd_1, d)
    #         loss_same += d

    # for i in range(24, 32):
    #     for j in range(32):
    #         d = distance(centerF, fea[i][j])
    #         maxd_2 = max(maxd_2, d)
    #         loss_same += d

    tmp1 = fea[torch.where(labels_map > 0)]
    tmp2 = fea[torch.where(labels_map == 0)]

    tmp1 = centerT-tmp1
    tmp1 = torch.mul(tmp1, tmp1)
    tmp1 = torch.sum(tmp1, dim=1)

    loss_same = loss_same+torch.sum(tmp1)

    tmp2 = centerF-tmp2
    tmp2 = torch.mul(tmp2, tmp2)
    tmp2 = torch.sum(tmp2, dim=1)

    loss_same = loss_same+torch.sum(tmp2)
    loss_same = loss_same/512

    # if iteration < 1800:
    #     loss = loss_dif
    # else:
    #     while loss_same > loss_dif:
    #         loss_same = loss_same/2
    #     loss = loss_same+loss_dif

    # while loss_same > loss_dif:
    #     loss_same = loss_same/2

    return loss_same


def distance(v1, v2):
    tmp = v1-v2
    tmp = torch.dot(tmp, tmp)
    tmp = torch.sum(tmp)
    return tmp


def data2seq(data: torch.Tensor, labels: torch.Tensor):

    labels = labels.squeeze()
    T_p = torch.where(labels > 0)[0]
    F_p = torch.where(labels == 0)[0]
    len_tp = T_p.shape[0]
    len_fp = F_p.shape[0]
    # x = data[:, :, 64:192, 64:192]
    # x = x.unfold(2, 32, 32)
    # x = x.unfold(3, 32, 32)
    # x = x.reshape(x.size(0), 3, -1, 32, 32)

    # x = x[:, :, torch.randperm(x.size(2)), :, :]
    # x = x.transpose(1, 2)  # x=[B,16,3,64,64]

    # seq_1 = torch.randint(0, 2, (1, 16))

    # # 设定开头为1
    # seq_1[0][0] = 1
    # seq_1[0][1] = 1
    # seq_1[0][2] = 1

    seq_num = 32

    # TF_set = torch.zeros((int(seq_num/2), 2))
    # TT_set = torch.zeros((int(seq_num/4), 2))
    # FF_set = torch.zeros((int(seq_num/4), 2))

    T_set = T_p[torch.randint(0, len_tp, (16, 4))]
    TF_setF = F_p[torch.randint(0, len_fp, (16, 4))]
    TF_set = torch.cat((TF_setT, TF_setF), 1)

    TT_set = T_p[torch.randint(0, len_tp, (8, 8))]
    FF_set = F_p[torch.randint(0, len_fp, (8, 8))]

    matrix_set = torch.cat([TF_set, TT_set, FF_set], 0)

    # matrix_set = torch.randint(0, x.size(0), (seq_num, 2))
    # seq_map = 0
    # # seq_map = seq_1.repeat((seq_num, 1))
    # for i in range(seq_num):
    #     seq_map[i] = matrix_set[i][seq_map[i]]

    labels_map = torch.zeros((32, 32))
    labels_map[0:16, 0:16] = 1
    labels_map[16:24, :] = 1
    # 将拼接对转成label对保存,然后将labels_map转化为跟一号位置的同label vector
    # labels_map = labels[matrix_set]

    # for i in range(seq_num):
    #     if labels_map[i][0] != 1:
    #         labels_map[i] = 1-labels_map[i]
    # # 生成拼接序列的随机顺序

    seq = torch.zeros((seq_num, 8, 4, 3, 32, 32))
    for i in range(seq_num):
        for j in range(8):
            seq[i][j] = data[matrix_set[i][j]]
    seq = seq.reshape(32, 32, 3, 32, 32)
    return labels_map, seq


'''
:method definition : DDP专用的模型加载方式
---------------------
:param model        需要被加载的模型
:param optimizer    需要被加载的优化器
:param path_list    保存的模型的路径[]
:param mode         选择的加载方式[0:加载训练最优的模型，1：加载最新的模型，2：加载制定模型]
:param index        制定模型的编号
'''

# TODO 维修现场


def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, path_list: str, mode: int, index: int, flag_t=False):

    if not os.path.exists(path_list[mode]):
        return 0, 0

    if flag_t:
        print("loading teacher model")
        whole = torch.load(path_list[mode])
        incomp_keys = model.load_state_dict(
            {k.replace('module.', ''): v for k, v in whole['model'].items()})
        print(incomp_keys)
        return
    print("loading student model")

    whole = torch.load(path_list[mode])
    # *加载模型参数
    incomp_keys = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in whole['model'].items()})
    print(incomp_keys)
    # *加载优化器参数
    opt_state = whole['opt']

    optimizer.load_state_dict(opt_state)

    # *加载其余参数
    epoch = whole['epoch']
    iteration = whole['iteration']
    # 常规变量就直接返回出去
    return epoch, iteration


'''
method definition :用于将训练过程中得到weight进行保存 
------------------
:param net[nn.Module]   过程中需要进行保存的model 
:param optimizer    训练过程中所使用的的优化器 
:param train_loss   当前的训练损失    
:param val_loss     当前的验证损失  
:param iteration    当前的迭代次数(迭代了多少个batch) 
:param batch_size   当前所采用的的batch大小
:param epoch        当前处于第几个eopch 
:param path         以上所有数据的存储路径
------------------
'''


def save_model_v2(model: nn.Module, optimizer: torch.optim.Optimizer,
                  train_loss: float, val_loss: float,
                  iteration: int, batch_size: int, epoch: int,
                  path: str):
    path = str(path)
    model_state_dict = model.state_dict()
    # optimizer_state_dict =optimizer.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()

    # for key in optimizer.:
    #     optimizer_state_dict[key] = optimizer_state_dict[key].cpu()
    state = dict(model=model_state_dict,
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch,
                 )
    torch.save(state, path)


'''
method definition:
-----------------
:param  net              需要验证的模型  
:param  device           运行设备编号
:param  val_loader       验证集的dataloader
:param  criterion        损失函数
:param  tb               tensorboard的实例化对象
:param  iteration        当前迭代次数
:param  tag              给定了，就是"val"
:param  loader_len_norm  每次验证集读取的个数
----------------
:return val_loss         模型在验证集上的loss 

'''


def validation_v1(net, val_loader_1, iteration, tb, tag):
    net.eval()
    acc_total1 = 0
    acc_total2 = 0
    val_num = 0
    val_loss_total = 0
    for val_data in tqdm(val_loader_1, desc='Validation', leave=False, total=len(val_loader_1)):
        (batch_data, batch_labels), batch_df = val_data
        val_batch_num = len(batch_labels)
        val_num = val_num+val_batch_num
        with torch.no_grad():
            val_acc1, val_acc2, val_pred = batch_forward_val(
                net, batch_data, batch_labels, batch_df, 0, iteration)
            val_loss_total = val_loss_total+val_loss
            acc_total1 = acc_total1+val_acc1*val_batch_num
            acc_total2 = acc_total2+val_acc2*val_batch_num

    acc_total1 = acc_total1/val_num
    acc_total2 = acc_total2/val_num
    val_loss_total = val_loss_total/len(val_loader_1)

    tb.add_scalar("val/"+tag+"acc1", acc_total1, iteration)
    tb.add_scalar("val/"+tag+"acc2", acc_total2, iteration)
    return acc_total2, val_loss_total


def validation_routine(net, val_loader, tag: str, iteration, tb, loader_len_norm: int = None):
    # switch to eval mode
    net.eval()

    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    val_acc = 0
    val_auc = 0
    val_labels_list1 = []
    val_pred_list1 = []
    val_labels_list2 = []
    val_pred_list2 = []
    val_labels_list3 = []
    val_pred_list3 = []
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

        (batch_data, batch_labels), batch_paths = val_data
        # 给定batch大小
        val_batch_num = len(batch_labels)

        with torch.no_grad():
            val_batch_loss, val_pred = batch_forward(net, batch_data,
                                                     batch_labels, batch_paths)
        # _, val_pred_index = torch.max(val_batch_pred, 1)
        # _, val_label_index = torch.max(batch_labels, 1)
        val_labels_list1.append(batch_labels.cpu())
        val_pred_list1.append(val_pred.cpu())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num
        # val_acc += val_batch_acc * val_batch_num
        # val_auc += auc * val_batch_num

    val_labels = np.concatenate(val_labels_list1).flatten()
    val_pred = np.concatenate(val_pred_list1).flatten()

    val_acc = accuracy_score(val_labels, val_pred)
    val_roc_auc = roc_auc_score(val_labels, val_pred)
    # Logging
    val_loss /= val_num
    # val_acc /= val_num
    # val_auc /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    tb.add_scalar('{}/acc'.format(tag), val_acc, iteration)
    tb.add_scalar('{}/auc'.format(tag), val_roc_auc, iteration)

    return val_loss


def main():

    args = config.config_test()
    mp.spawn(dist_train,  nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
