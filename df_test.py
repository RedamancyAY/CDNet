import numpy as np
import pandas as pd
from PIL import Image
import cv2

celebdf_df_path = '/mnt/8T/hou/celeb-df/faces/celeb_df.pkl'
FFPP_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'

df = pd.read_pickle(FFPP_df_path)

# df_1 = df[(df['class'] == 'Celeb-synthesis')]
df_1 = df[df['label'] == 1]

for i in range(200, 300):
    tmp = df_1.iloc[i]

    print(tmp)

    # path = '/mnt/8T/hou/celeb-df/faces/'+tmp.name
    path = '/mnt/8T/hou/FFPP/faces/output/'+tmp.name

    pic = Image.open(path)
    pic = np.array(pic)

    size = int((tmp.right-tmp.left)/8)

    pic_1 = cv2.rectangle(pic, (tmp.left, tmp.top),
                          (tmp.right, tmp.bottom), (240, 54, 45), 3)

    pic_1 = cv2.rectangle(pic_1, (tmp.kp1x-size, tmp.kp1y-size),
                          (tmp.kp1x+size, tmp.kp1y+size), (255, 110, 184), 3)

    pic_1 = cv2.circle(pic_1, (tmp.kp1x, tmp.kp1y),
                       2, (229, 20, 0))

    pic_1 = cv2.rectangle(pic_1, (tmp.kp2x-size, tmp.kp2y-size),
                          (tmp.kp2x+size, tmp.kp2y+size), (255, 110, 184), 3)

    pic_1 = cv2.circle(pic_1, (tmp.kp2x, tmp.kp2y),
                       2, (229, 20, 0))

    pic_1 = cv2.rectangle(pic_1, (tmp.kp3x-size, tmp.kp3y-size),
                          (tmp.kp3x+size, tmp.kp3y+size), (255, 110, 184), 3)

    pic_1 = cv2.circle(pic_1, (tmp.kp3x, tmp.kp3y),
                       2, (229, 20, 0))

    pic_1 = cv2.rectangle(pic_1, (tmp.kp4x-size, tmp.kp4y-size),
                          (tmp.kp4x+size, tmp.kp4y+size), (255, 110, 184), 3)

    pic_1 = cv2.circle(pic_1, (tmp.kp4x, tmp.kp4y),
                       2, (229, 20, 0))

    cv2.imwrite('/mnt/8T/hou/pic/'+str(i)+'.jpg', pic_1)

# print(tmp)
