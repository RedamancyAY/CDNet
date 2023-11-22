from itertools import count
from turtle import forward, update
from cv2 import CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH, _InputArray_CUDA_HOST_MEM
from torch import nn
import torch
from pytorch_wavelets import DWTForward
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class WaveGen(nn.Module):

    def __init__(self):
        super().__init__()
        self.DWT1 = DWTForward(J=1, mode='symmetric', wave='db2')
        self.DWT2 = DWTForward(J=1, mode='symmetric', wave='coif1')
        self.DWT3 = DWTForward(J=1, mode='symmetric', wave='sym2')
        self.DWTs = [self.DWT1, self.DWT2, self.DWT3]

    '''
    :method description: gain the DWT descompose of X, only keep the high frequency coefficients
    ----------------------
    :param: x
    ----------------------
    :return 1 layer wavelet decompose of x 
    '''

    def DWT(self, x: torch.Tensor):
        size = int(x.shape[-1]/2)
        batch = x.shape[0]
        channel = x.shape[1]
        # 当channel==3时，即channel与小波种类不相关
        if channel == 3:

            x_H = [f(x)[1][0][..., :size, :size] for f in self.DWTs]

            x_L = [f(x)[0][..., :size, :size] for f in self.DWTs]

        # 当channel!=3,按channel拆分后再进行小波
        else:
            channel_pw = int(channel/3)
            splits = torch.split(x, channel_pw, 1)
            x_H = [f(splits[index])[1][0][..., :size, :size]
                   for index, f in enumerate(self.DWTs)]
            x_L = [f(splits[index])[0][..., :size, :size]
                   for index, f in enumerate(self.DWTs)]

        x_H = [torch.reshape(item, (batch, -1, size, size)) for item in x_H]
        x_L = [torch.reshape(item, (batch, -1, size, size)) for item in x_L]
        x_H = torch.cat(tuple(x_H), dim=1)
        x_L = torch.cat(tuple(x_L), dim=1)

        return x_L, x_H

    def features(self, x: torch.Tensor):
        # get the Tensor first, Tensor size=[B,C,D(HVD),H,W]
        x_low, x_high = self.DWT(x)
        # x = self.DWT(x)

        return x_low, x_high

    def forward(self, x):
        x_low, x_high = self.features(x)
        return x_low, x_high


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        # torch.cuda.empty_cache()
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        fea = self.features(x)

        return x


class Xception_1(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_1, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 128, 2, 1, start_with_relu=True, grow_first=True)

        self.block3 = Block(
            128, 256, 3, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(
            256, 256, 3, 1, start_with_relu=True, grow_first=True)
        # self.block5 = Block(
        #     256, 512, 3, 1, start_with_relu=True, grow_first=True)
        # self.block6 = Block(
        #     512, 512, 3, 1, start_with_relu=True, grow_first=True)

        self.block5 = Block(
            256, 512, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(512, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)

        # do relu here

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        fea = self.features(x)

        return x


class b_f(nn.Module):
    def __init__(self, seqnum):
        super().__init__()
        self.DWT = WaveGen()

    def features(self, input):
        return

    def forward(self, input):
        x = input  # input = [seqnum*16,3,64,64]
        x_L, x_H = self.DWT(x)
        x_LL, x_LH = self.DWT(x_L)
        x_HL, x_HH = self.DWT(x_H)
        x_LLL, x_LLH = self.DWT(x_LL)
        x_LHL, x_LHH = self.DWT(x_LH)
        x_HLL, x_HLH = self.DWT(x_HL)
        x_HHL, x_HHH = self.DWT(x_HH)

        H_1 = x_H
        H_2 = torch.cat([x_LH, x_HH], 1)
        H_3 = torch.cat([x_LHH, x_LLH, x_HLH, x_HHH], 1)

        return [H_1, H_2, H_3]


class b_s(nn.Module):
    def __init__(self, seqnum):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(27)
        self.bn2 = nn.BatchNorm2d(108)
        self.bn3 = nn.BatchNorm2d(432)

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(400, 400)
        self.block1 = Block(
            91, 182, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            182, 182, 3, 1, start_with_relu=False, grow_first=True)
        self.block3 = Block(
            182, 182, 3, 1, start_with_relu=False, grow_first=True)

        self.block4 = Block(
            290, 580, 2, 2, start_with_relu=False, grow_first=True)
        self.block5 = Block(
            580, 580, 3, 1, start_with_relu=False, grow_first=True)
        self.block6 = Block(
            580, 580, 3, 1, start_with_relu=False, grow_first=True)

        self.block7 = Block(
            1012, 1012, 2, 2, start_with_relu=False, grow_first=True)
        self.block8 = Block(
            1012, 1012, 2, 2, start_with_relu=False, grow_first=True)
        self.block9 = Block(
            1012, 600, 3, 1, start_with_relu=False, grow_first=True)
        self.block10 = Block(
            600, 400, 3, 1, start_with_relu=False, grow_first=True)

    def features(self, input):
        return

    def forward(self, input):
        (x, fea_f) = input  # [seq_num*16,3,64,64]
        [H_1, H_2, H_3] = fea_f
        H_1 = self.bn1(H_1)
        H_2 = self.bn2(H_2)
        H_3 = self.bn3(H_3)

        x = self.conv1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = torch.cat((x, H_1), 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.cat((x, H_2), 1)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = torch.cat((x, H_3), 1)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class b_s_2(nn.Module):
    def __init__(self, seqnum):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(27)
        self.bn2 = nn.BatchNorm2d(108)
        self.bn3 = nn.BatchNorm2d(432)

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(400, 400)
        self.block1 = Block(
            91, 182, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            182, 182, 3, 1, start_with_relu=False, grow_first=True)
        self.block3 = Block(
            182, 182, 3, 1, start_with_relu=False, grow_first=True)

        self.block4 = Block(
            290, 580, 2, 2, start_with_relu=False, grow_first=True)
        self.block5 = Block(
            580, 580, 3, 1, start_with_relu=False, grow_first=True)
        self.block6 = Block(
            580, 580, 3, 1, start_with_relu=False, grow_first=True)

        self.block7 = Block(
            1012, 1012, 2, 2, start_with_relu=False, grow_first=True)
        self.block8 = Block(
            1012, 1012, 2, 2, start_with_relu=False, grow_first=True)
        self.block9 = Block(
            1012, 600, 3, 1, start_with_relu=False, grow_first=True)
        self.block10 = Block(
            600, 400, 3, 1, start_with_relu=False, grow_first=True)

    def features(self, input):
        return

    def forward(self, input):
        (x, fea_f) = input  # [seq_num*16,3,64,64]
        [H_1, H_2, H_3] = fea_f
        H_1 = self.bn1(H_1)
        H_2 = self.bn2(H_2)
        H_3 = self.bn3(H_3)

        x = self.conv1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = torch.cat((x, H_1), 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.cat((x, H_2), 1)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = torch.cat((x, H_3), 1)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class match_net(nn.Module):
    def __init__(self, seqnum):
        super().__init__()
        self.seqnum = seqnum

    def features(self, input):
        return

    def forward(self, input: torch.Tensor):
        x = input
        x = F.normalize(x, 2, 1)
        x = x.reshape(self.seqnum, 16, 200)
        x[:, 3:, :].detach()
        x_head = x[:, 0:3, :]

        x_t = x.transpose(1, 2)
        sim = torch.bmm(x_head, x_t)
        # print(sim[5])
        sim = (sim+1)/2
        sim = torch.mean(sim, 1)
        # print(sim[5])
        # _, seq = torch.min(sim, 1)
        # seq = (sim[:, 0, :] >= sim[:, 1, :]).long()

        return sim


class model_1(nn.Module):
    def __init__(self, seqnum):
        super().__init__()
        self.branch1 = b_s(seqnum)
        self.branch2 = b_f(seqnum)
        self.matchNet = match_net(seqnum)
        self.xce = Xception_1()
        self.centerT = torch.zeros(200)
        self.centerF = torch.zeros(200)
        self.count = 0
        # self.xce.load_state_dict(
        #     model_zoo.load_url(model_urls['xception']))
        self.linear = nn.Linear(2048, 200)
        self.lr = 0

    def features(self, input):
        return

    def update_centers(self, fea, centerT, centerF, labels_map, lr):
        self.count = self.count+1
        centerT = centerT.detach()
        centerF = centerF.detach()
        lr_center = 0.01

        tmp1 = fea[torch.where(labels_map > 0)[0]]
        tmp2 = fea[torch.where(labels_map == 0)[0]]
        # tmp1_p = torch.sort(tmp1)[1]
        # tmp2_p = torch.sort(tmp2)[1]

        # len_1 = tmp1.shape[0]
        # len_2 = tmp2.shape[0]

        centerT_currentB = torch.mean(tmp1, 0)
        centerF_currentB = torch.mean(tmp2, 0)

        # if torch.sum(centerT) == 0:
        #     centerT = centerT_currentB
        #     centerF = centerF_currentB
        # else:
        #     centerT = (1-lr_center)*centerT+lr_center*centerT_currentB
        #     centerF = (1-lr_center)*centerF+lr_center*centerF_currentB
        # 最多每次移动0.01长的center距离
        if torch.sum(centerT) == 0:
            centerT = centerT_currentB
            centerF = centerF_currentB
        else:
            centerT = (1-lr_center)*centerT+lr_center * \
                (1)*centerT_currentB
            centerF = (1-lr_center)*centerF+lr_center * \
                (1)*centerF_currentB

            # if self.count < 1800:
            #     centerT = (1-lr_center)*centerT+lr_center * \
            #         (1)*centerT_currentB
            #     centerF = (1-lr_center)*centerF+lr_center * \
            #         (1)*centerF_currentB
            # else:
            #     distanceT = centerT-centerT_currentB
            #     distanceT = torch.dot(distanceT, distanceT)
            #     distanceT = distanceT.detach()

            #     distanceF = centerF-centerF_currentB
            #     distanceF = torch.dot(distanceF, distanceF)
            #     distanceF = distanceF.detach()

            #     distanceTF = centerT-centerF
            #     distanceTF = torch.dot(distanceTF, distanceTF)
            #     distanceTF = distanceTF.detach()

            #     centerT = (1-lr_center)*centerT+lr_center * \
            #         (distanceTF/distanceT)*centerT_currentB
            #     centerF = (1-lr_center)*centerF+lr_center * \
            #         (distanceTF/distanceF)*centerF_currentB

        return centerT, centerF

    def forward(self, input: torch.Tensor, labels_map, lr):
        x = input
        x = x.reshape(-1, 3, 32, 32)
        fea = self.xce.features(x)
        fea_f = self.branch2(x)
        # fea = self.branch1((x, fea_f))
        fea = self.linear(fea)

        self.centerT, self.centerF = self.update_centers(
            fea, self.centerT, self.centerF, labels_map, lr)
        centers = self.centerT, self.centerF
        # pred = self.matchNet(fea)

        return fea, centers


class model_2(nn.Module):
    def __init__(self, seqnum):
        super().__init__()
        self.branch1 = b_s_2(seqnum)
        self.xce = Xception_1()
        self.centerT = torch.zeros(200)
        self.centerF = torch.zeros(200)
        self.count = 0
        # self.xce.load_state_dict(
        #     model_zoo.load_url(model_urls['xception']))
        self.linear = nn.Linear(1024, 2)
        self.lr = 0

    def features(self, input):
        return

    def forward(self, input: torch.Tensor):
        x = input
        # x = x.reshape(-1, 12, 32, 32)
        x0 = x[:, 0, :, :, :]
        x1 = x[:, 1, :, :, :]
        x2 = x[:, 2, :, :, :]
        x3 = x[:, 3, :, :, :]
        fea0 = self.xce.features(x0)
        fea1 = self.xce.features(x1)
        fea2 = self.xce.features(x2)
        fea3 = self.xce.features(x3)
        fea = torch.cat((fea0, fea1, fea2, fea3), dim=1)
        # fea = self.branch1((x, fea_f))
        fea = self.linear(fea)

        # pred = self.matchNet(fea)

        return fea


class cuttingTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 1)
        self.model_2 = model_2(64)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    def get_normalizer(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input: torch.Tensor):
        x = input

        x = self.model_2(x)
        # seqs_set =
        return x


class globalProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    def get_normalizer(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 1)
        self.model_1 = model_1(64)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    def get_normalizer(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images: torch.Tensor, cuttings: torch.Tensor, labels_map, lr):

        x, centers = self.model_1(x, labels_map, lr)

        # seqs_set =
        return x, centers
