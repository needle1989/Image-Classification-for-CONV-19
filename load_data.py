import os
import torch
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# load csv file
f = open('trainData/slice-level/Slice_level_label.csv')
tag = list(csv.reader(f))
# root path for train set
root_cap = "trainData/slice-level/Cap/"
root_p = "trainData/slice-level/Covid-19/"
root_n = "trainData/subject-level/Non-infected/"


# return root path for every image in this folder
def file_name(file_dir):
    ct_img = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file[0] != '.':
                ct_img.append(root + "/" + file)
    return ct_img


def default_loader(path):
    return Image.open(path).convert('RGB')


class LoadData(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        p_sample = 0
        while p_sample < 55:
            p_sample = p_sample + 1
            p = file_name(root_p + tag[p_sample][0])
            for num in range(len(p)):
                imgs.append((p[num], int(tag[p_sample][num + 1])))
        cap_sample = 55
        while cap_sample < 80:
            cap_sample = cap_sample + 1
            cap = file_name(root_cap + tag[cap_sample][0])
            for num in range(len(cap)):
                cap_tag = int(tag[cap_sample][num + 1])
                if cap_tag == 1:
                    cap_tag = cap_tag + 1
                imgs.append((cap[num], cap_tag))
        n_sample = 100
        while n_sample < 200:
            n_sample = n_sample + 1
            normal = file_name(root_n + 'normal' + n_sample)
            for num in range(len(normal)):
                imgs.append((normal[num], 0))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)