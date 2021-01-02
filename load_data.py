import os
import torch
from PIL import Image
from torch.utils.data import dataset

class TrainSet(dataset):
    def __int__(self):
        cap = "trainData/subject-level/Cap"
        patient = "trainData/subject-level/Covid-19"
        normal = "trainData/subject-level/Non-infected"
