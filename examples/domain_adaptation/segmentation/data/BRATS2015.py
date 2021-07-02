import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import random as r
import SimpleITK as sitk
import os

img_size = 120      #original img size is 240*240
smooth = 1 

import glob

def create_data(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + '**/*{}*.mha'.format(mask), recursive=True)
    print('Processing---', mask)

    imgs = []
    for file in files:
        patient = file.split('/')[-2]
        file_name = file.split('/')[-1][0:-4]
        img = io.imread(file, plugin='simpleitk')
        # img = trans.resize(img, resize, mode='constant')

        # 保存图片不做处理
        # if label:
        #     #img[img == 4] = 1       #turn enhancing tumor into necrosis
        #     #img[img != 1] = 0       #onl
        #     y left enhancing tumor + necrosis
        #     #img[img != 0] = 1       #Region 1 => 1+2+3+4 complete tumor
        #     img = img.astype('float32')
        # else:
        #     img = (img-img.mean()) / img.std()      #normalization => zero mean   !!!care for the std=0 problem

        img = img[50:130]

        path1 = '/home/weiyuhua/TransferBed/examples/domain_adaptation/segmentation/data/BRATS2015'
        if not os.path.exists(os.path.join(path1, mask, patient)):
            os.makedirs(os.path.join(path1, mask, patient))

        # write image/label list
        path2 = '/home/weiyuhua/TransferBed/examples/domain_adaptation/segmentation/data/BRATS2015/image_list'
        txt_file = os.path.join(path2, '{}.txt'.format(mask))

        with open(txt_file, "a") as f:
            for i in range(img.shape[0]):
                io.imsave(os.path.join(path1, mask, patient, '{}_slide_{}.png'.format(file_name, i)), img[i])
                f.write(os.path.join(mask, patient, '{}_slide_{}.png'.format(file_name, i) + '\n'))

        # img = np.expand_dims(img, axis=0)
        # imgs.append(img)
    # imgs = np.concatenate(imgs, axis=0)
    # imgs = np.array(imgs)
    # name = mask + 'npy'
    # np.save(name, imgs)
    # print('{} Saved'.format(name))


def n4itk(img):         #must input with sitk img object
    img = sitk.Cast(img, sitk.sitkFloat32)
    img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
    corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
    return corrected_img    



# catch all T1c.mha
data_dir = '/home/weiyuhua/TransferBed/examples/domain_adaptation/segmentation/data/mhafiles/BRATS2015_Training/HGG/'

create_data(data_dir, 'Flair', label=False, resize=(155,img_size,img_size))
create_data(data_dir, 'T1', label=True, resize=(155,img_size,img_size))
create_data(data_dir, 'T1c', label=True, resize=(155,img_size,img_size))
create_data(data_dir, 'T2', label=True, resize=(155,img_size,img_size))
create_data(data_dir, 'OT', label=True, resize=(155,img_size,img_size))

#%%
# catch BRATS2017 Data
# create_data('/home/andy/Brain_tumor/BRATS2017/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/', '**/*_flair.nii.gz', label=False, resize=(155,img_size,img_size))
# create_data('/home/andy/Brain_tumor/BRATS2017/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/', '**/*_GlistrBoost_ManuallyCorrected.nii.gz', label=True, resize=(155,img_size,img_size))


#%%
# load numpy array data
x = np.load('/home/andy/x_{}.npy'.format(img_size))
y = np.load('/home/andy/y_{}.npy'.format(img_size))

'''
animation
'''
import matplotlib.animation as animation
def animate(pat, gifname):
    # Based on @Zombie's code
    fig = plt.figure()
    anim = plt.imshow(pat[50])
    def update(i):
        anim.set_array(pat[i])
        return anim,
    
    a = animation.FuncAnimation(fig, update, frames=range(len(pat)), interval=50, blit=True)
    a.save(gifname, writer='imagemagick')
    
#animate(pat, 'test.gif')
