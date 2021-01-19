import os
import torch
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import SimpleITK as SITK
import glob
import numpy as np
from PIL import Image
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# cap data:000-138 Covid-19 data:000-288 normal data:000-203
root_cap = "trainData/subject-level/Cap/"
root_p = "trainData/subject-level/Covid-19/"
root_n = "trainData/subject-level/Non-infected/"


# return root path for every image in this folder
def file_name(file_dir):
    ct_img = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file[0] != '.':
                ct_img.append(root + "/" + file)
    return ct_img


def save_array_as_nii_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the output file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = SITK.GetImageFromArray(data)
    if reference_name is not None:
        img_ref = SITK.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    SITK.WriteImage(img, filename)


allImg = []
allImg = np.zeros([512, 512, 512])
p = file_name(root_cap + 'cap001')
print(p[1])
for i in range(len(p)):
    single_image_name = p[i]
    img_as_img = Image.open(single_image_name)
    # img_as_img.show()
    img_as_np = np.asarray(img_as_img)
    allImg[i, :, :] = img_as_np
# save_array_as_nii_volume(allImg, './testImg.nii.gz')
print(np.shape(allImg))
print(allImg[1, :, :])

# FA_org = SITK.load('./testImg.nii.gz')
# FA_data = FA_org.get_data()
