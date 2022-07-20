import torch.nn as nn
import torch.nn.functional as F

class Denoising_Model(nn.Module):
    """
    Model definition.
    Y is residual
    X is clean data
    Y'is noisy data =>
    Y = Y'- X,
    X = Y' - Y
    Layers:
        1 Conv1D layer+ReLU,
        18 Conv1D layers+BatchNorm+RelU,
        1 Conv1D layer
    """

    def __init__(self, nLayers=10):
        super(Denoising_Model, self).__init__()
        # DnCNN architecture
        self.repeated_layers = nLayers - 2
        self.conv_list = nn.ModuleList()
        for nLayer in range(self.repeated_layers):
            self.conv_list.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
            self.conv_list.append(nn.BatchNorm1d(64))
            self.conv_list.append(nn.ReLU())
            if nLayer > 0 and nLayer % 4 == 0:
                self.conv_list.append(nn.Dropout(0.5))

        self.ConvFirst = nn.Conv1d(in_channels=80, out_channels=64, kernel_size=3, padding=1)
        self.ConvLast = nn.Conv1d(in_channels=64, out_channels=80, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = F.relu(self.ConvFirst(inputs))

        for layer in self.conv_list:
            x = layer(x)

        x = self.ConvLast(x)

        return x