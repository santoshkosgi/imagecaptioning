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


class Decoder(nn.Module):
    """
    Here we define the Decoder class. LSTM.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seqlength=20):
        """
        :param embed_size: Size of the output of embedding layer.
        :param hidden_size: Size of the hidden layer.
        :param vocab_size: Number of distinct words in the data.
        :param num_layers: Number of stacked LSTM layers.
        :param max_seqlength: Length of the maximum sequence to be considered.
        """
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seqlength

    def forward(self, features, captions, lengths):
        """

        :param features: Encoder output for each image. Dimension: batch_size * embed_size
        :param captions: captions for each image. Dimension: batch_size * max_length_of_caption
        :param lengths: length of the each caption.
        :return: Softmax output for each time stamp.
        """
        embeddings = self.embed(captions)
        # Passing o/p of encoder to the lstm during the first time stamp.
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        lstm_out, (ht, ct) = self.lstm(packed)
        outputs = self.linear(lstm_out.data)
        return outputs

    def sample(self, features, states= None):
        input = features.unsqueeze(1)
        caption_indes = []

        seq = 0
        while seq < self.max_seg_length:
            hidden, states = self.lstm(input, states)
            output = self.linear(hidden.squeeze(1))
            _, predicted = output.max(1)
            caption_indes.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            input = inputs.unsqueeze(1)
            seq += 1
        caption_indes = torch.stack(caption_indes, 1)
        return caption_indes

