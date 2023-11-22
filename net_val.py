

import torch
import pandas as pd

from model import xception, models2

from efficientnet_pytorch import EfficientNet

p1 = "/media/ubuntu/hou/multicard_CNET/weights/binclass/net-CNet_traindb-ff-c23-720-140-140_face-scale_patchSize-299_cuttingsSize-64_seed-43_note-NT_1/it000003.pth"
# p2 = "/mnt/8T/hou/multicard_gen/weights/binclass/net-WholeNet_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-Gtest2_02_2/it001000.pth"

model1 = torch.load(p1)
# model2 = torch.load(p2)


# df = pd.DataFrame(columns=['part', 'change'])
# count = 0
# for item1, item2 in zip(model1['model'].items(), model2['model'].items()):

#     part = item1[0]
#     if("module.judge.j_linear.weight" not in part):
#         continue
#     change = abs((item1[1]-item2[1])/item1[1])
#     df.loc[count] = [part, change.numpy()]
#     count = count + 1

# df.to_csv("2.csv")
# print(1)
model = model1['model']
model_x = EfficientNet.from_name('efficientnet-b4')

pytorch_total_params = sum(p.numel() for p in model_x.parameters())
trainable_pytorch_total_params = sum(
    p.numel() for p in model_x.parameters() if p.requires_grad)

# torch.save(model, '1.pth')
for item in model:
    print(model[item].shape)
# total = sum([for item in model])
# print('  + Number of params: %.2fM' % (total / 1e6))
