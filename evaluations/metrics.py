import cv2
import numpy as np
import os
from PIL import Image
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.measure import shannon_entropy as shannon_entropy
import lpips
import torch
import torchvision.transforms as transforms
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.io
from evaluations.UQI import UQI
from evaluations.VIF import compare_vifp
# from UQI import UQI
# from VIF import compare_vifp

alpha_p = np.arange(0.2, 10, 0.001)
alpha_r_p = scipy.special.gamma(2.0 / alpha_p) ** 2 / (
        scipy.special.gamma(1.0 / alpha_p) * scipy.special.gamma(3. / alpha_p))


def estimate_aggd_params(x):
    x_left = x[x < 0]
    x_right = x[x >= 0]
    stddev_left = math.sqrt((np.sum(x_left ** 2) / x_left.size))
    stddev_right = math.sqrt((np.sum(x_right ** 2) / x_right.size))

    if stddev_right == 0:
        return 1, 0, 0
    r_hat = np.sum(np.abs(x)) ** 2 / (x.size * np.sum(x ** 2))
    y_hat = stddev_left / stddev_right  # gamma hat
    R_hat = r_hat * (y_hat ** 3 + 1) * (y_hat + 1) / ((y_hat ** 2 + 1) ** 2)

    pos = np.argmin((alpha_r_p - R_hat) ** 2)
    alpha = alpha_p[pos]
    beta_left = stddev_left * math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    beta_right = stddev_right * math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    return alpha, beta_left, beta_right


def compute_nss_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)
    features.extend([alpha, (beta_left + beta_right) / 2])

    for x_shift, y_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        img_pair_products = img_norm * np.roll(np.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0 / alpha) / gamma(1.0 / alpha))
        features.extend([alpha, eta, beta_left, beta_right])

    return features


def norm(img, sigma=7 / 6):
    mu = gaussian_filter(img, sigma, mode='nearest', truncate=2.2)
    sigma = np.sqrt(np.abs(gaussian_filter(img * img, sigma, mode='nearest', truncate=2.2) - mu * mu))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm


def niqe(image):
    if image.ndim == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image

    model_mat = scipy.io.loadmat(os.path.split(__file__)[0] + '/resources/mvg_params.mat')
    model_mu = model_mat['mean']
    model_cov = model_mat['cov']

    features = None
    h, w = img.shape
    quantized_h = max(h // 96, 1) * 96
    quantized_w = max(w // 96, 1) * 96

    quantized_img = img[:quantized_h, :quantized_w]
    img_scaled = quantized_img
    for scale in [1, 2]:
        if scale != 1:
            img_scaled = cv2.resize(quantized_img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        img_norm = norm(img_scaled.astype(float))
        scale_features = []
        block_size = 96 // scale
        for block_col in range(img_norm.shape[0] // block_size):
            for block_row in range(img_norm.shape[1] // block_size):
                block_features = compute_nss_features(
                    img_norm[block_col * block_size:(block_col + 1) * block_size,
                    block_row * block_size:(block_row + 1) * block_size])
                scale_features.append(block_features)

        if features is None:
            features = np.vstack(scale_features)
        else:
            features = np.hstack([features, np.vstack(scale_features)])

    features_mu = np.mean(features, axis=0)
    features_cov = np.cov(features.T)

    pseudoinv_of_avg_cov = np.linalg.pinv((model_cov + features_cov) / 2)
    niqe_quality = math.sqrt((model_mu - features_mu).dot(pseudoinv_of_avg_cov.dot((model_mu - features_mu).T)))

    return niqe_quality


def ave(lis):
    s = 0
    total_num = len(lis)
    for i in lis:
        s = s + i
    return s / total_num


def PSNR(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


path1 = r'C:\Users\64883\Desktop\qxs_h1\test_latest\images\fake_B/'
path2 = r'C:\Users\64883\Desktop\qxs_h1\test_latest\images\real_B/'

# path1 = r'E:\pix2pix_test\test_latest\images/'
# path2 = r'E:\pix2pix_test\test_latest\images/'

f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
list_en = []
list_lpips = []
list_niqe_real = []
list_niqe_fake = []
list_uqi = []
list_vif = []

for i in range(0, 1096):
    img_a = Image.open(path1 + format(str(i), '0>5s') + '.png')
    img_b = Image.open(path2 + format(str(i), '0>5s') + '.png')
    # img_a = Image.open(path1 + format(str(i), '0>5s') + '_fake_B.png')
    # img_b = Image.open(path2 + format(str(i), '0>5s') + '_real_B.png')
    img_a = np.array(img_a)
    img_b = np.array(img_b)
    img_ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    img_gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    # psnr_num = compare_psnr(img_ga, img_gb, data_range=255)
    psnr_num = PSNR(img_ga, img_gb)
    # print(psnr_num)
    ssim_num = compare_ssim(img_ga, img_gb, data_range=255)
    # print(ssim_num)
    en_num = shannon_entropy(img_ga, base=2)

    # LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex', version=0.1)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg',
                              version=0.1)  # closer to "traditional" perceptual loss, when used for optimization

    result = cv2.imread(path1 + format(str(i), '0>5s') + '.png')
    GT = cv2.imread(path2 + format(str(i), '0>5s') + '.png')
    # result = cv2.imread(path1 + format(str(i), '0>5s') + '_fake_B.png')
    # GT = cv2.imread(path2 + format(str(i), '0>5s') + '_real_B.png')
    test1_res = result
    test1_label = GT
    transf = transforms.ToTensor()
    test1_label = transf(test1_label)
    test1_res = transf(test1_res)
    test1_ress = test1_res.to(torch.float32).unsqueeze(0)
    test1_labell = test1_label.to(torch.float32).unsqueeze(0)
    lpips_loss = loss_fn_alex(test1_ress, test1_labell)

    # print(niqe(result), niqe(GT))
    niqe_loss_real = niqe(GT)
    niqe_loss_fake = niqe(result)

    uqi_sum = UQI(img_ga, img_gb)
    vif_sum = compare_vifp(img_ga, img_gb)

    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    list_en.append(en_num)
    list_lpips.append(lpips_loss)
    list_niqe_real.append(niqe_loss_real)
    list_niqe_fake.append(niqe_loss_fake)
    list_uqi.append(uqi_sum)
    list_vif.append(vif_sum)

print('平均PSNR:', np.mean(list_psnr))
print('平均SSIM:', np.mean(list_ssim))
print('平均en:', np.mean(list_en))
print('平均lpips:', ave(list_lpips))
print('平均niqe_real:', ave(list_niqe_real))
print('平均niqe_fake:', ave(list_niqe_fake))
print('平均uqi:', ave(list_uqi))
print('平均vif:', ave(list_vif))

# entropy越大,信息越丰富
# LPIPS用于度量两张图像之间的差别,越高意味着图片与原图更多不同,越低意味着与原图更相似
# PSNR经常用作图像压缩等领域中信号重建质量的测量方法
# SSIM可以衡量图片的失真程度,也可以衡量两张图片的相似程度
# NIQE值越小,图像质量越好
# FID分数越低代表两组图像越相似,或者说二者的统计量越相似
# UQI通过检查图像的结构信息来衡量图像之间的相似程度,UQI可以在[0,1]的范围内,值越接近1表示相似度越高。
# VIF测量基于两个样本之间的共同信息的数量来评估两个图像的质量和相似性,VIF值越高,相似性越显著。
# python -m pytorch_fid C:\Users\64883\Desktop\ref_cy\kaist_daytime\exp C:\Users\64883\Desktop\ref_cy\kaist_daytime\ref
