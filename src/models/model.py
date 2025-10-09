from src.models.cifar10_alexnet import CIFAR10_AlexNet
from src.models.cifar100_alexnet import CIFAR100_AlexNet
from src.models.mnist_dnn import Mnist_DNN
from src.models.mnist_cnn import Mnist_CNN
from src.models.fmnist_cnn import FMnist_CNN
from src.models.svhn_vgg16 import SVHN_VGG16
from src.models.svhn_cnn import SVHN_CNN
from src.models.emnist_cnn import EMNISTCNN
from src.models.emnist_dnn import EMNISTDNN
from src.models.svhn_alexnet import svhn_AlexNet
import torch
import torch.nn as nn
import numpy as np


def choose_model(options):
    model_name = str(options['model_name']).lower()
    torch.manual_seed(2025)
    if model_name == 'mnist_dnn':
        # for name, param in Mnist_CNN().named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        return Mnist_DNN()
    if model_name == 'mnist_cnn':
        return Mnist_CNN()
    elif model_name == 'fmnist_cnn':
        return FMnist_CNN()
    elif model_name == 'emnist_cnn':
        return EMNISTCNN()
    elif model_name == 'emnist_dnn':
        return EMNISTDNN()
    elif model_name == 'cifar10_alexnet':
        return CIFAR10_AlexNet()
    elif model_name == 'svhn_vgg16':
        return SVHN_VGG16()
    elif model_name == 'svhn_cnn':
        return SVHN_CNN()
    elif model_name == 'svhn_alexnet':
        return svhn_AlexNet()

    elif model_name == 'cifar100_alexnet':
        return CIFAR100_AlexNet()



