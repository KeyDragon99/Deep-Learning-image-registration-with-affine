import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()

        self.conv_layer1 = nn.Sequential(nn.Conv2d(2, 64, 3, 2, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU())
        self.conv_layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU())

        # Regressor for the 3 * 2 affine matrix
        self.transl_fc_layer = nn.Sequential(
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

        self.scale_fc_layer = nn.Sequential(
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

        self.rot_fc_layer = nn.Sequential(
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

        # Initialize the weights/bias with identity transformation
        self.transl_fc_layer[5].weight.data.zero_()
        self.transl_fc_layer[5].bias.data.copy_(torch.tensor([0., 0.], dtype=torch.float))

        self.scale_fc_layer[5].weight.data.zero_()
        self.scale_fc_layer[5].bias.data.copy_(torch.tensor([1., 1.], dtype=torch.float))

        self.rot_fc_layer[5].weight.data.zero_()
        self.rot_fc_layer[5].bias.data.copy_(torch.tensor([0., 0.], dtype=torch.float))

    def forward(self, x):

        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)

        out = out.contiguous().view(-1, 256 * 32 * 32)
        transl = self.transl_fc_layer(out)
        scale = self.scale_fc_layer(out)
        rot = self.rot_fc_layer(out)

        return transl.squeeze(), scale.squeeze(), rot.squeeze()
