import os
import shutil
import tqdm

old_path = r'E:\colorization\M3FD_day\trainA/'
new_path = r'E:\colorization\M3FD_day\testA/'

files = os.listdir(old_path)
# for i in tqdm.tqdm(range(len(files))):
#     print(files[i][:-6])
for i in tqdm.tqdm(range(len(files))):
    if int(files[i][:-4]) % 20 == 0:
        old_file_path = old_path + '/' + files[i]
        new_file_path = new_path + '/' + files[i]
        shutil.move(old_file_path, new_file_path)
