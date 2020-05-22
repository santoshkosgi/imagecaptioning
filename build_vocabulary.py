"""
This function converts the captions to word indices.
"""
import argparse
import nltk
from collections import Counter
import pickle
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class Vocabulary(object):
    """
    This class builds the vocabulary, string to index mapping.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def main(args):
    train_images = open(args.trainimages_path, "r")

    train_images = [line.strip() for line in list(train_images)]

    counter = Counter()

    for caption_label in open(args.caption_path, "r"):
        image_name, caption = caption_label.split(maxsplit=1)
        image_name = image_name.split("#")[0]
        # Only considering the images in training.
        if image_name in train_images:
            caption = caption.strip()
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
    words = [word for word in counter if counter[word] > args.threshold]

    vocabulary = Vocabulary()
    vocabulary.add_word('<pad>')
    vocabulary.add_word('<start>')
    vocabulary.add_word('<end>')
    vocabulary.add_word('<unk>')

    for word in words:
        vocabulary.add_word(word)

    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocabulary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainimages_path', type=str,
                        default='Flickr8k_text/Flickr_8k.trainImages.txt',
                        help='path for Images Annotation file')
    parser.add_argument('--caption_path', type=str,
                        default='Flickr8k_text/Flickr8k.lemma.token.txt',
                        help='path for Images Annotation file')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=0,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
