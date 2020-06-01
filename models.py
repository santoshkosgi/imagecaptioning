import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import math
beamsearch_n = 10


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

    def sample(self, features, end_token_index, states= None):
        input = features.unsqueeze(1)
        caption_indes = []

        # input - input to the lstm
        # states - (h,c) of lstm
        # [] - output sequence till now
        # 1 - probability of the sequence predicted till now.
        queue = [(input, states, [], 0, 0, True), "level"]
        # This stores scores of all possible outcomes of a level.
        # When processing of a level is done, top beamsearch_n entries from these are populated to queue.
        all_scores_level = []
        seq = 0
        while True:
            if queue[0] == "level":
                # End of a level
                queue.pop(0)
                queue = sorted(all_scores_level, key=lambda k: k[4], reverse=True)[:beamsearch_n]
                queue.append("level")
                seq += 1
                if not (seq < self.max_seg_length):
                    break
                all_scores_level = []
                continue
            input, states, caption_list, prob, norm_prob, shall_continue = queue.pop(0)
            if not shall_continue:
                all_scores_level.append((input, states, caption_list, prob, norm_prob, shall_continue))
                continue
            hidden, states = self.lstm(input, states)
            output = self.linear(hidden.squeeze(1))
            output = nn.functional.softmax(output, dim=1)

            # To consider beamsearch_n top words.
            values, indices = torch.topk(output, beamsearch_n)
            idx = 0
            while idx < beamsearch_n:
                to_continue = True
                if indices[0][idx] == end_token_index:
                    to_continue = False
                input_ = self.embed(indices[0][idx]).unsqueeze(0).unsqueeze(0)
                all_scores_level.append((input_, states, caption_list + [indices[0][idx]],
                                         prob + math.log(values[0][idx]),
                                         (prob + math.log(values[0][idx]))/(len(caption_list) + 1), to_continue))
                idx += 1
            _, predicted = output.max(1)
            caption_indes.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            input = inputs.unsqueeze(1)

        sorted_captions_list = sorted(all_scores_level, key=lambda k: k[4], reverse=True)

        caption_indes = torch.tensor(sorted_captions_list[0][2]).unsqueeze(0)
        return caption_indes

    def find_prob_of_actual_caption(self, features, capion_index_list, states=None):
        """
        This function finds the probability of the actual captions.
        :param features: Image representation
        :param states: (h,c) of LSTM
        :return:
        """
        input = features.unsqueeze(1)
        prob = 0
        for caption in capion_index_list:
            hidden, states = self.lstm(input, states)
            output = self.linear(hidden.squeeze(1))
            output = nn.functional.softmax(output, dim=1)
            prob = prob + math.log(output[0][caption])
            inputs = self.embed(torch.tensor([caption]))  # inputs: (batch_size, embed_size)
            input = inputs.unsqueeze(1)

        print("Probability of actual caption is ", prob/len(capion_index_list))

    def sample_eval(self, features, end_token_index, states= None):
        """
        This function generates captions for the multiple images at the same time.
        :param features:
        :param end_token_index:
        :param states:
        :return:
        """

