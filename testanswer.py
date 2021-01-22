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
from load_data import LoadData
import numpy as np
import time

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

path="covid_classify_model_8.model"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint)
model.eval()

def predict_image(image_path):
    print("Prediction in progress")
    image = Image.open(image_path)
    if image.mode == 'L':
        image = image.convert('RGB')

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    index = output.data.numpy().argmax()

    return index

if __name__ == "__main__":

    imagepath = "testData/"
    path=[
        os.path.join(x)
        for x in os.listdir("testData/T000") if x[0] != '.'
    ]

    # run prediction function annd obtain prediccted class index
    for j in range(len(path)):
        index = predict_image(imagepath+path[j])
        print("Predicted Class ", index,"image name",imagepath+path[j])