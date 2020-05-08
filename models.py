import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    """
    Here we define different layers in encode.
    """
    def __init__(self, embed_size):
        """

        :param embed_size: Size of the image embedding.
        """
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Do not want to update resnet weights.
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.linear(features))
        return features




