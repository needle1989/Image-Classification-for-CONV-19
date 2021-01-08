import os
import torch
import torch.nn as nn
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

# load csv file
f = open('trainData/slice-level/Slice_level_label.csv')
tag = list(csv.reader(f))
# root path for train set
root_cap = "trainData/slice-level/Cap/"
root_p = "trainData/slice-level/Covid-19/"


# return root path for every image in this folder
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
                cap_tag = int(tag[cap_sample][num + 1])
                if cap_tag == 1:
                    cap_tag = cap_tag + 1
                imgs.append((cap[num], cap_tag))
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


'''
class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()


        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleNet,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128*16*16,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output
'''
# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(512, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 16

train_set = LoadData(transform=train_transformations);

validation_split = .2
shuffle_dataset = True
random_seed = 42
set_size = len(train_set)
indices = list(range(set_size))
split = int(np.floor(validation_split * set_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=valid_sampler)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)

if cuda_avail:
    model.cuda()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "covid_classify_model_{}.model".format(epoch))
    print("Checkpoint saved")


def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            total += labels.size(0)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all training images
        train_acc = train_acc / total
        train_loss = train_loss / total

        # Save the model if the test acc is greater than our current best
        if train_acc > best_acc:
            save_models(epoch)
            best_acc = train_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} ".format(epoch, train_acc, train_loss))


if __name__ == "__main__":
    train(200)
