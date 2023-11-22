import imp
from itertools import count
from turtle import forward, update
import cv2
from cv2 import CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH, _InputArray_CUDA_HOST_MEM, dct
from torch import nn
import torch
from pytorch_wavelets import DWTForward
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import torch_dct

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}

DCT_table = [0, 1, 32, 2, 33, 64, 3, 34, 65, 96, 4, 35, 66, 97, 128, 5, 36, 67, 98, 129, 160, 6, 37, 68, 99, 130, 161, 192, 7, 38, 69, 100, 131, 162, 193, 224, 8, 39, 70, 101, 132, 163, 194, 225, 256, 9, 40, 71, 102, 133, 164, 195, 226, 257, 288, 10, 41, 72, 103, 134, 165, 196, 227, 258, 289, 320, 11, 42, 73, 104, 135, 166, 197, 228, 259, 290, 321, 352, 12, 43, 74, 105, 136, 167, 198, 229, 260, 291, 322, 353, 384, 13, 44, 75, 106, 137, 168, 199, 230, 261, 292, 323, 354, 385, 416, 14, 45, 76, 107, 138, 169, 200, 231, 262, 293, 324, 355, 386, 417, 448, 15, 46, 77, 108, 139, 170, 201, 232, 263, 294, 325, 356, 387, 418, 449, 480, 16, 47, 78, 109, 140, 171, 202, 233, 264, 295, 326, 357, 388, 419, 450, 481, 512, 17, 48, 79, 110, 141, 172, 203, 234, 265, 296, 327, 358, 389, 420, 451, 482, 513, 544, 18, 49, 80, 111, 142, 173, 204, 235, 266, 297, 328, 359, 390, 421, 452, 483, 514, 545, 576, 19, 50, 81, 112, 143, 174, 205, 236, 267, 298, 329, 360, 391, 422, 453, 484, 515, 546, 577, 608, 20, 51, 82, 113, 144, 175, 206, 237, 268, 299, 330, 361, 392, 423, 454, 485, 516, 547, 578, 609, 640, 21, 52, 83, 114, 145, 176, 207, 238, 269, 300, 331, 362, 393, 424, 455, 486, 517, 548, 579, 610, 641, 672, 22, 53, 84, 115, 146, 177, 208, 239, 270, 301, 332, 363, 394, 425, 456, 487, 518, 549, 580, 611, 642, 673, 704, 23, 54, 85, 116, 147, 178, 209, 240, 271, 302, 333, 364, 395, 426, 457, 488, 519, 550, 581, 612, 643, 674, 705, 736, 24, 55, 86, 117, 148, 179, 210, 241, 272, 303, 334, 365, 396, 427, 458, 489, 520, 551, 582, 613, 644, 675, 706, 737, 768, 25, 56, 87, 118, 149, 180, 211, 242, 273, 304, 335, 366, 397, 428, 459, 490, 521, 552, 583, 614, 645, 676, 707, 738, 769, 800, 26, 57, 88, 119, 150, 181, 212, 243, 274, 305, 336, 367, 398, 429, 460, 491, 522, 553, 584, 615, 646, 677, 708, 739, 770, 801, 832, 27, 58, 89, 120, 151, 182, 213, 244, 275, 306, 337, 368, 399, 430, 461, 492, 523, 554, 585, 616, 647, 678, 709, 740, 771, 802, 833, 864, 28, 59, 90, 121, 152, 183, 214, 245, 276, 307, 338, 369, 400, 431, 462, 493, 524, 555, 586, 617, 648, 679, 710, 741, 772, 803, 834, 865, 896, 29, 60, 91, 122, 153, 184, 215, 246, 277, 308, 339, 370, 401, 432, 463, 494, 525, 556, 587, 618, 649, 680, 711, 742, 773, 804, 835, 866, 897, 928, 30, 61, 92, 123, 154, 185, 216, 247, 278, 309, 340, 371, 402, 433, 464, 495, 526, 557, 588, 619, 650, 681, 712, 743, 774, 805, 836, 867, 898, 929, 960, 31, 62, 93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465, 496, 527, 558, 589, 620, 651, 682, 713, 744, 775, 806, 837, 868,
             899, 930, 961, 992, 63, 94, 125, 156, 187, 218, 249, 280, 311, 342, 373, 404, 435, 466, 497, 528, 559, 590, 621, 652, 683, 714, 745, 776, 807, 838, 869, 900, 931, 962, 993, 95, 126, 157, 188, 219, 250, 281, 312, 343, 374, 405, 436, 467, 498, 529, 560, 591, 622, 653, 684, 715, 746, 777, 808, 839, 870, 901, 932, 963, 994, 127, 158, 189, 220, 251, 282, 313, 344, 375, 406, 437, 468, 499, 530, 561, 592, 623, 654, 685, 716, 747, 778, 809, 840, 871, 902, 933, 964, 995, 159, 190, 221, 252, 283, 314, 345, 376, 407, 438, 469, 500, 531, 562, 593, 624, 655, 686, 717, 748, 779, 810, 841, 872, 903, 934, 965, 996, 191, 222, 253, 284, 315, 346, 377, 408, 439, 470, 501, 532, 563, 594, 625, 656, 687, 718, 749, 780, 811, 842, 873, 904, 935, 966, 997, 223, 254, 285, 316, 347, 378, 409, 440, 471, 502, 533, 564, 595, 626, 657, 688, 719, 750, 781, 812, 843, 874, 905, 936, 967, 998, 255, 286, 317, 348, 379, 410, 441, 472, 503, 534, 565, 596, 627, 658, 689, 720, 751, 782, 813, 844, 875, 906, 937, 968, 999, 287, 318, 349, 380, 411, 442, 473, 504, 535, 566, 597, 628, 659, 690, 721, 752, 783, 814, 845, 876, 907, 938, 969, 1000, 319, 350, 381, 412, 443, 474, 505, 536, 567, 598, 629, 660, 691, 722, 753, 784, 815, 846, 877, 908, 939, 970, 1001, 351, 382, 413, 444, 475, 506, 537, 568, 599, 630, 661, 692, 723, 754, 785, 816, 847, 878, 909, 940, 971, 1002, 383, 414, 445, 476, 507, 538, 569, 600, 631, 662, 693, 724, 755, 786, 817, 848, 879, 910, 941, 972, 1003, 415, 446, 477, 508, 539, 570, 601, 632, 663, 694, 725, 756, 787, 818, 849, 880, 911, 942, 973, 1004, 447, 478, 509, 540, 571, 602, 633, 664, 695, 726, 757, 788, 819, 850, 881, 912, 943, 974, 1005, 479, 510, 541, 572, 603, 634, 665, 696, 727, 758, 789, 820, 851, 882, 913, 944, 975, 1006, 511, 542, 573, 604, 635, 666, 697, 728, 759, 790, 821, 852, 883, 914, 945, 976, 1007, 543, 574, 605, 636, 667, 698, 729, 760, 791, 822, 853, 884, 915, 946, 977, 1008, 575, 606, 637, 668, 699, 730, 761, 792, 823, 854, 885, 916, 947, 978, 1009, 607, 638, 669, 700, 731, 762, 793, 824, 855, 886, 917, 948, 979, 1010, 639, 670, 701, 732, 763, 794, 825, 856, 887, 918, 949, 980, 1011, 671, 702, 733, 764, 795, 826, 857, 888, 919, 950, 981, 1012, 703, 734, 765, 796, 827, 858, 889, 920, 951, 982, 1013, 735, 766, 797, 828, 859, 890, 921, 952, 983, 1014, 767, 798, 829, 860, 891, 922, 953, 984, 1015, 799, 830, 861, 892, 923, 954, 985, 1016, 831, 862, 893, 924, 955, 986, 1017, 863, 894, 925, 956, 987, 1018, 895, 926, 957, 988, 1019, 927, 958, 989, 1020, 959, 990, 1021, 991, 1022, 1023]


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


class xce_cuttings(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(xce_cuttings, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 1, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 128, 3, 1, start_with_relu=True, grow_first=True)

        self.block3 = Block(
            128, 256, 2, 1, start_with_relu=True, grow_first=True)
        self.block4 = Block(
            256, 256, 3, 1, start_with_relu=True, grow_first=True)

        self.conv3 = SeparableConv2d(256, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)

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

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        fea = self.features(x)

        return fea


class xce_images(nn.Module):

    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(xce_images, self).__init__()

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
            128, 128, 2, 1, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            128, 128, 1, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            256, 256, 2, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            256, 256, 1, 2, start_with_relu=True, grow_first=True)

        self.conv3 = SeparableConv2d(256, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)

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

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        fea = self.features(x)

        return fea


class f_cuttings(nn.Module):
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(f_cuttings, self).__init__()

    def features(self, x):
        x = dct.dct_2d(x)
        # x = cv2.dct()
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)
        # x = x.numpy()
        # x = x[DCT_table]
        return x

    def forward(self, x):
        fea = self.features(x)

        return fea


class f_ex(nn.Module):
    def __init__(self):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(f_ex, self).__init__()

    def features(self, x):
        x = torch_dct.dct_2d(x)
        x = x[:, :, :64, :64]
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)
        x = x.cpu().numpy()
        x = x[:, DCT_table]
        x = x.reshape(-1, 128, 8)
        x = x.mean(axis=2)
        x = torch.from_numpy(x).cuda()

        return x

    def forward(self, x):
        fea = self.features(x)

        return fea


class feaProcess(nn.Module):
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(feaProcess, self).__init__()
        self.l1 = nn.Linear(2432, 1)
        self.l2 = nn.Linear(200, 1)
        self.relu = nn.ReLU()

    def features(self, x):
        x = self.l1(x)
        # x = self.relu(x)
        # x = self.l2(x)

        return x

    def forward(self, x):
        fea = self.features(x)

        return fea

# backup 2022.10.11
# class CNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         """ Constructor
#         Args:
#             num_classes: number of classes
#         """
#         super(CNet, self).__init__()

#         self.xce_images = xce_images()
#         self.xce_cuttings = xce_cuttings()
#         self.f_ex = f_ex()
#         # self.f_p = feaProcess()
#         self.ln = nn.Linear(1920, 200)
#         self.fc = nn.Linear(200, 1)

#     def get_trainable_parameters(self):
#         a = filter(lambda p: p.requires_grad, self.parameters())
#         return a

#     def get_normalizer(self):
#         return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def features(self, images, cuttings):
#         cuttings = torch.reshape(cuttings, (-1, 3, 64, 64))

#         x_i = self.xce_images(images)
#         x_c = self.xce_cuttings(cuttings)
#         x_c = torch.reshape(x_c, (-1, 1024))
#         f_i = self.f_ex(images)
#         f_c = self.f_ex(cuttings)
#         f_c = torch.reshape(f_c, (-1, 512))
#         fea = torch.cat((x_i, x_c, f_i, f_c), dim=1)
#         fea = self.ln(fea)
#         return fea

#     def forward(self, images, cuttings):
#         fea = self.features(images, cuttings)
#         pred = self.fc(fea)
#         # fea = fea.reshape(-1, 4, 1)
#         # fea = torch.mean(fea, dim=1)
#         return pred


class CNet(nn.Module):
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(CNet, self).__init__()

        self.xce_images = xce_images()
        self.xce_cuttings = xce_cuttings()
        self.f_ex = f_ex()
        # self.f_p = feaProcess()
        self.ln = nn.Linear(1920, 200)
        self.fc = nn.Linear(200, 1)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    def get_normalizer(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def features(self, images, cuttings):
        cuttings = torch.reshape(cuttings, (-1, 3, 64, 64))

        x_i = self.xce_images(images)
        x_c = self.xce_cuttings(cuttings)
        x_c = torch.reshape(x_c, (-1, 1024))
        f_i = self.f_ex(images)
        f_c = self.f_ex(cuttings)
        f_c = torch.reshape(f_c, (-1, 512))
        fea = torch.cat((x_i, x_c, f_i, f_c), dim=1)
        fea_o = self.ln(fea)
        return fea_o

    def forward(self, images, cuttings):
        fea = self.features(images, cuttings)
        # o = self.fc(fea)
        # fea = fea.reshape(-1, 4, 1)
        # fea = torch.mean(fea, dim=1)
        return fea
