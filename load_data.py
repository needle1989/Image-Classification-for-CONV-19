import os
import torch
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader

f = open('trainData/slice-level/Slice_level_label.csv')
tag = list(csv.reader(f))
root_cap = "trainData/slice-level/Cap/"
root_p = "trainData/slice-level/Covid-19/"


def file_name(file_dir):
    ct_img = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
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
            print(len(p))
            for num in range(len(p)):
                print(p[num])
                print(tag[p_sample][num + 1])
                imgs.append((p[num], int(tag[p_sample][num + 1])))
        cap_sample = 55
        while cap_sample < 80:
            cap_sample = cap_sample + 1
            cap = file_name(root_cap + tag[cap_sample][0])
            print(len(cap))
            for num in range(len(cap)):
                print(cap[num])
                print(tag[cap_sample][num + 1])
                imgs.append((cap[num], int(tag[cap_sample][num + 1])))
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


ct = file_name(root_cap + tag[56][0])
print(ct[0])
print(tag[56][0])
print(tag[56][1])
count = 55
while count < 80:
    count = count + 1
    ct = file_name(root_cap + tag[count][0])
    print(len(ct))
    for i in range(len(ct)):
        print(ct[i])
        print(tag[count][i + 1])
