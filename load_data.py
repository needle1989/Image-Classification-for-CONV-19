import os
import torch
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# load csv file
f = open('trainData/slice-level/Slice_level_label.csv')
tag = list(csv.reader(f))
# root path for train set
root_cap = "trainData/slice-level-v2/slice-level/Cap/"
root_p = "trainData/slice-level-v2/slice-level/Covid-19/"
root_normal = "trainData/subject-level/Non-infected/"


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
            # print(len(p))
            for num in range(len(p)):
                # print(p[num])
                # print(tag[p_sample][num + 1])
                imgs.append((p[num], int(tag[p_sample][num + 1])))
        cap_sample = 55
        while cap_sample < 80:
            cap_sample = cap_sample + 1
            cap = file_name(root_cap + tag[cap_sample][0])
            # print(len(cap))
            for num in range(len(cap)):
                # print(cap[num])
                # print(tag[cap_sample][num + 1])
                cap_tag = int(tag[cap_sample][num + 1])
                if cap_tag == 1:
                    cap_tag = cap_tag + 1
                print(cap_tag)
                imgs.append((cap[num], cap_tag))

        non_paths = [
            os.path.join(x)
            for x in os.listdir("trainData/subject-level/Non-infected") if x[0] != '.'
        ]
        for j in range(len(non_paths)):
            if j > 60:
                break
            p = file_name(root_normal + non_paths[j])
            for i in range(len(p)):
                imgs.append((p[i], 0))
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