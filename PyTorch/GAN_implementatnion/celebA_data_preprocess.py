import os
import matplotlib.pyplot as plt
# from scipy.misc import imresize
from skimage.transform import resize
import shutil

# 注意文件运行时候的路径
shutil.rmtree('../data/resized_celebA/celebA')
# root path depends on your computer
root = r'H:/data/img_align_celeba/'
save_root = '../data/resized_celebA/'

resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = resize(img, (resize_size, resize_size))
    # img = scipy.misc.imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)