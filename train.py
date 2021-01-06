import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from load_data import LoadData

# 图像的预处理
train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    batchsize=32; #先试试之后可改

    # 数据读取和预处理
    trainset= LoadData(transform=train_transformer)
    print(trainset.__len__())
    # 构建dataloader
    trainloader=DataLoader(trainset,batch_size=batchsize,shuffle=True)

    # 模型
    

    # 损失函数

    # 优化器

    # 训练
    # 这里在参考代码那里是有一个额外的文件做训练，然后我这里主要是看看数据有没有导进来就先用输出了，之后改改
    for epoch in range(2):
        for batch_index,batch_samples in enumerate(trainloader):
            # 将数据读取出来
            inputs,labels=batch_samples

            # 输出
            print(epoch,batch_index,"images",inputs,"labels",labels);
