"""
This script has methods to load the encoder and decoder trained models and given a image
generates the caption for it.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from PIL import Image
from imagecaptioning.models import Encoder
from imagecaptioning.models import Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from imagecaptioning.build_vocabulary import Vocabulary
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_image(path, transform=None):
    """
    This function loads the image a specific path and returns it
    :param path: Location of the image file
    :return: Image object.
    """
    image = Image.open(path).convert('RGB')

    image = image.resize((256, 256), Image.ANTIALIAS)

    if transform is not None:
        image = transform(image)

    # Convert image to dimension 1*256*256
    image = image.unsqueeze(0)

    return image


def main(args):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # .eval to specify its evaluation.
    encoder = Encoder(args.embed_size).eval()

    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers, args.attention_dim).eval()

    # Load saved model
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    image = load_image(args.image, transform)

    features = encoder(image)

    sampled_ids = decoder.sample_greedy(features, vocab.word2idx['<start>'])

    # decoder.find_prob_of_actual_caption(features, [vocab.word2idx[word.lower()] for word in args.caption])

    sampled_ids = sampled_ids[0].numpy()
    sampled_caption = []

    for id in sampled_ids:
        word = vocab.idx2word[id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # decoder.find_prob_of_actual_caption(features, [vocab.word2idx[word.lower()] for word in sampled_caption])

    # Print out the image and the generated caption
    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--caption', type=str, nargs='+', required=False, help='actual caption of the image')
    parser.add_argument('--encoder_path', type=str, default='models/encoder.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='Data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--attention_dim', type=int, default=512, help='Dimension of the attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
