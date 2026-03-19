from itertools import repeat
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2.cv2 as cv2

NUM_THREADS = os.cpu_count()


def calc_channel_var(img_path):  # 计算标准差的辅助函数
    img = np.array(Image.open(img_path).convert('L')) / 255.0
    h, w = img.shape
    mean = img.sum() / (h * w)
    channel_var = np.sum((img - mean) ** 2, axis=(0, 1))
    return channel_var


if __name__ == '__main__':
    train_path = Path(r'C:\Users\64883\Desktop\RoadScene-master\cropinfrared')
    train_path1 = Path(r'C:\Users\64883\Desktop\RoadScene-master\crop_LR_visible')
    img_f = list(train_path.rglob('*.jpg'))
    img_v = list(train_path1.rglob('*.jpg'))
    # img_v = cv2.cvtColor(img_v, cv2.COLOR_RGB2GRAY)
    n = len(img_f)

    for path1, path2 in zip(img_f, img_v):
        # print(x)
        var_fir = calc_channel_var(path1)
        var_vis = calc_channel_var(path2)
        # print(path1, min(var_fir, var_vis))
        if min(var_fir, var_vis) < 15:
            os.remove(path1)
            os.remove(path2)
    # print("R_var is %f, G_var is %f, B_var is %f" % (var[0], var[1], var[2]))
