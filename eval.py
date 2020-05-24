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
from imagecaptioning.dataloader_dev import get_loader
from torchvision import transforms
from imagecaptioning.build_vocabulary import Vocabulary
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import nltk


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

    with open(args.caption_path) as json_file:
        devdatajson = json.load(json_file)

    data_loader = get_loader(args.image_dir, devdatajson, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # .eval to specify its evaluation.
    encoder = Encoder(args.embed_size).eval()

    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).eval()

    # Load saved model
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    blue_score = 0
    for i, (images, captions) in enumerate(data_loader):
        images = images
        captions = captions
        features = encoder(images)
        # Iterating through each of the image to get the captions of it.
        for i in range(features.shape[0]):
            feature = features[i][:].unsqueeze(0)
            sampled_ids = decoder.sample(feature, vocab.word2idx['<end>'])
            sampled_ids = sampled_ids[0].numpy()
            sampled_caption = []
            for id in sampled_ids:
                word = vocab.idx2word[id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            sentence = sentence.split(" ")[1:-1]
            BLEUscore = nltk.translate.bleu_score.sentence_bleu(split_captions(captions[i]), sentence)
            blue_score += BLEUscore
    print("Bleu score is ", blue_score/len(devdatajson))

def split_captions(captions):
    """
    This function  converts each caption to a list. For blue score computation
    :param captions: list of captions for an image
    :return: Caption in bleu score format.
    """
    result = []
    for caption in captions:
        result.append(caption.split(" "))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='Flicker8k_Dataset', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='Data/devdata.json',
                        help='path for train annotation json file')
    parser.add_argument('--encoder_path', type=str, default='models/encoder.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='Data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    main(args)
