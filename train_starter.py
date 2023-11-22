from distutils.command.config import config

import CNet_train
import CNet_train2
import EX3

from config import config

import torch.multiprocessing as mp


# *train a teacher at a time
def singleT_train():
    args = config.config_test()
    mp.spawn(CNet_train.dist_train,  nprocs=args.gpus, args=(args,))
    return 0

# *train four teachers in sequence


def multiCNet_train():
    tagnotes = ['DF_1', 'F2F_1', 'FS_1', 'NT_1']

    for i in range(3, 4):
        args = config.config_test()
        args.tagnote = tagnotes[i]
        args.trainIndex = i
        mp.spawn(CNet_train.dist_train,  nprocs=args.gpus, args=(args,))
    return 0


def multiCNet_train2():

    models_paths = ["/media/ubuntu/hou/multicard_CNET/weights/binclass/net-CNet_traindb-ff-c23-720-140-140_face-scale_patchSize-299_cuttingsSize-64_seed-43_note-DF_1/it000020.pth",
                    "/media/ubuntu/hou/multicard_CNET/weights/binclass/net-CNet_traindb-ff-c23-720-140-140_face-scale_patchSize-299_cuttingsSize-64_seed-43_note-F2F_1/it000020.pth",
                    "/media/ubuntu/hou/multicard_CNET/weights/binclass/net-CNet_traindb-ff-c23-720-140-140_face-scale_patchSize-299_cuttingsSize-64_seed-43_note-FS_1/it000020.pth",
                    "/media/ubuntu/hou/multicard_CNET/weights/binclass/net-CNet_traindb-ff-c23-720-140-140_face-scale_patchSize-299_cuttingsSize-64_seed-43_note-NT_1/it000020.pth"

                    ]
    tagnotes = ['DF_2', 'F2F_2', 'FS_2', 'NT_2']
    for i in range(0, 4):
        args = config.config_test()
        args.tagnote = tagnotes[i]
        args.trainIndex = i
        args.mp = models_paths[i]
        mp.spawn(CNet_train2.dist_train,  nprocs=args.gpus, args=(args,))
    return 0


def multiCNet_train3():
    tagnotes = ['ab_sr0.0', 'ab_sr0.1',
                'ab_sr0.2', 'ab_sr0.3', 'ab_sr0.4', 'ab_sr0.5']
    mrs = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    srs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(0, 6):
        args = config.config_test()
        args.tagnote = tagnotes[i]
        # args.trainIndex = i
        args.moveRate = mrs[2]
        args.shrinkRate = srs[i]
        args.epochs = 20
        mp.spawn(EX3.dist_train,  nprocs=args.gpus, args=(args,))
    return 0


def multiCJELS_train():
    tagnotes = ['DF_vKD2', 'F2F_vKD2', 'FS_vKD2']
    # tagnotes = ['DF_normalKD', 'F2F_normalKD', 'FS_normalKD']
    net_t_paths = ['/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-DF_T1/last.pth',
                   '/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-F2F_T1/last.pth',
                   '/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-FS_T1/last.pth'
                   ]
    for i in range(2, 3):
        args = config.config_train()
        args.tagnote = tagnotes[i]
        args.net_t_path = net_t_paths[i]
        args.trainIndex = i
        mp.spawn(CJEL_train.dist_EL,  nprocs=args.gpus, args=(args,))

    return 0


def modelgen():
    tagnote = ''
    netpath = ''
    args = config.config_gen()
    mp.spawn(CJEL_gen.dist_gen,  nprocs=args.gpus, args=(args,))


if __name__ == '__main__':

    # pick your train preference
    # singleT_train()
    # multiT_train()
    multiCNet_train3()
