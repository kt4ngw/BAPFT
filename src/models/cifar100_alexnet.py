import torch
import torch.nn as nn
import torch.nn.functional as F
num_classes = 20

class CIFAR100_AlexNet(nn.Module):
    def __init__(self, num_classes=20, init_weights=False):
        super(CIFAR100_AlexNet, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
                                          torch.nn.BatchNorm2d(64),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 192, kernel_size=4, stride=1, padding=1),
                                          torch.nn.BatchNorm2d(192),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0))
        
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384),
                                          torch.nn.ReLU(inplace=True))
        
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True))
        
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.avgpool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(output_size=(3, 3)))
        
        self.fc1 = torch.nn.Sequential(
                                       torch.nn.Linear(256 * 3 * 3, 1024),
                                       torch.nn.ReLU(inplace=True))
        
        self.fc2 = torch.nn.Sequential(
                                       torch.nn.Linear(1024, 1024),
                                       torch.nn.ReLU(inplace=True))
        
        self.linear = torch.nn.Linear(1024, num_classes)
        
        self.feature_extractor = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
                                               self.avgpool, nn.Flatten(), self.fc1, self.fc2)

        self.classifier = nn.Sequential(self.linear)


        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        feature = self.feature_extractor(x)
        pred = self.classifier(feature)

        return feature, pred
    
    def get_model_size(self, ):
        total_params = 0
        for name, param in CIFAR100_AlexNet().named_parameters():
            layer_params = param.numel()
            total_params += layer_params
            # print(f"{name}: {layer_params} parameters")
        total_params_kb = (total_params * 4) / 1024 / 1024
        return total_params_kb
    
    def feature2logit(self, x):
        return self.classifier(x)
    
if __name__ == '__main__':
    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in CIFAR100_AlexNet().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")