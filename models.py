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

    def forward(self, images):
        # Do not want to update resnet weights.
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.shape[0], -1)
        return features


class Attention(nn.Module):
    """
    Here we define the Attention Model class.
    Intuition of Attention layer is to find the weights of each pixel of the image to look for next.

    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: Dimension of the encoded image. Batch_size * Image size
        :param decoder_dim: Hidden dimension of the decoder image. Batch_size * hidden_dim_size
        :param attention_dim: Dimension of the attention layer.
        """
        super(Attention, self).__init__()
        self.image_attention = nn.Linear(encoder_dim, attention_dim)
        self.hidden_attention = nn.Linear(decoder_dim, attention_dim)
        self.attention_layer = nn.Linear(attention_dim, encoder_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded_image, hidden_state):
        """
        This function accepts encoded image and hidden state of the previous lstm cell and
        returns the attention output.
        :param encoded_image: encoded image
        :param hidden_state: Hidden state
        :return:
        """
        attention_image = self.image_attention(encoded_image)
        attention_hidden = self.hidden_attention(hidden_state)
        attention_params = self.softmax(self.attention_layer(self.relu(attention_hidden + attention_image)))
        return encoded_image * attention_params

class Decoder(nn.Module):
    """
    Here we define the Decoder class. LSTM.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, attention_dim, max_seqlength=20, encoder_dim=2048):
        """
        :param embed_size: Size of the output of embedding layer.
        :param hidden_size: Size of the hidden layer.
        :param vocab_size: Number of distinct words in the data.
        :param num_layers: Number of stacked LSTM layers.
        :param max_seqlength: Length of the maximum sequence to be considered.
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(encoder_dim + embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seqlength
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.init_h = nn.Linear(encoder_dim, hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, hidden_size)  # linear layer to find initial cell state of LSTMCell


    def init_hidden(self, features):
        """
        This function initialises the initial states of LSTM cell.
        :param features: Encoded image of dimension.
        :return:
        """
        h = self.init_h(features)
        c = self.init_c(features)
        return h,c

    def forward(self, features, captions, lengths):
        """

        :param features: Encoder output for each image. Dimension: batch_size * embed_size
        :param captions: captions for each image. Dimension: batch_size * max_length_of_caption
        :param lengths: length of the each caption.
        :return: Softmax output for each time stamp.
        """
        # Call the attention layer.

        lengths = [l - 1 for l in lengths]

        h, c = self.init_hidden(features)

        features_attention = self.attention(features, h)

        embeddings = self.embed(captions)

        max_length = lengths[0]
        # So now LSTM should run for 30 sequences. At each sequence following for loop selects
        # images which have that length and passes to LSTM Cell
        outputs = torch.zeros((features.shape[0], max_length, self.vocab_size))
        for i in range(max_length):
            batch_len = sum([l > i for l in lengths])
            lstm_features = torch.cat((features_attention[:batch_len, :], embeddings[:batch_len, i, :]), dim=1)
            h, c = self.lstm(lstm_features, (h[:batch_len], c[:batch_len]))
            pred = self.linear(h)
            outputs[:batch_len, i, :] = pred
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

