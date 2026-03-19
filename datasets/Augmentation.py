# 导入必要的库
import os
import torch
from torchvision import transforms
from PIL import Image

# 定义一些参数
input_dir = r"E:\colorization\train_val\training\train_VIS_HR"  # 输入图片所在的文件夹
output_dir = r"E:\colorization\train_val\training\trainB"  # 输出图片要保存的文件夹
crop_size = (320, 320)  # 裁剪后的图片大小，可以随意指定
stride_x = 128  # x方向滑动的大小，可以随意指定
stride_y = 128  # y方向滑动的大小，可以随意指定

# 创建输出文件夹，如果不存在的话
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 定义一个放大函数，输入一张图片和放大倍数，返回放大后的图片（注意：这里使用了torchvision.transforms.Resize方法）
def resize_image(image, scale):
    width, height = image.size  # 获取原始图片宽度和高度
    new_width = int(width * scale)  # 计算新的宽度（向下取整）
    new_height = int(height * scale)  # 计算新的高度（向下取整）
    return transforms.Resize((new_height, new_width))(image)  # 放大并返回


# 定义一个裁剪函数，输入一张图片和裁剪位置，返回裁剪后的图片（注意：这里使用了PIL.Image.crop方法）
def crop_image(image, x, y):
    left = x * stride_x  # 左上角x坐标
    upper = y * stride_y  # 左上角y坐标
    right = left + crop_size[0]  # 右下角x坐标
    lower = upper + crop_size[1]  # 右下角y坐标
    return image.crop((left, upper, right, lower))  # 裁剪并返回


# 遍历输入文件夹中的所有图片文件名
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(
            ".bmp"):  # 只处理jpg或png格式的图片，可以根据需要修改或添加其他格式
        image_path = os.path.join(input_dir, filename)  # 拼接完整的图片路径

        image = Image.open(image_path)  # 用PIL库打开图片

        image = resize_image(image, scale=2.0)  # 调用放大函数将图片放大两倍

        width, height = image.size  # 获取放大后的图片宽度和高度

        num_x = (width - crop_size[0]) // stride_x + 1  # 计算x方向可以裁剪多少张图片（向下取整）
        num_y = (height - crop_size[1]) // stride_y + 1  # 计算y方向可以裁剪多少张图片（向下取整）

        for x in range(num_x):  # 遍历x方向上所有可能的位置（从0开始）
            for y in range(num_y):  # 遍历y方向上所有可能的位置（从0开始）
                cropped_image = crop_image(image, x, y)  # 调用裁剪函数得到裁剪后的图片

                output_filename = f"{filename}_{x}_{y}.png"  # 根据原始文件名和位置生成输出文件名

                output_path = os.path.join(output_dir, output_filename)  # 拼接完整的输出路径

                cropped_image.save(output_path)  # 保存裁剪后的图片到输出路径

                print(f"Saved {output_filename} to {output_path}")  # 打印保存信息
