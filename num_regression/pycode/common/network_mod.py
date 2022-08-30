import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, dropout_rate, use_pretrained_vgg=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained_vgg)
        self.cnn = vgg.features

        dim_fc_in = 512*(resize//32)*(resize//32)
        self.fc = nn.Sequential(
            nn.Linear(dim_fc_in, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(18, 2)
        )

    def initializeWeights(self):
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                list_fc_param_value.append(param_value)
            
        return list_cnn_param_value, list_fc_param_value

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x