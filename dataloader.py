"""
This function has methods to load the images and their captions.
Preprocess the images to the format needed by resnet 101 and split the data to train, test and
validation.
"""
import os
import torch
import torchvision.datasets as dset
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import nltk
from build_vocabulary import Vocabulary
import pickle
import json


class FlickrDataset(data.Dataset):

    def __init__(self, root, trainingdatajson, vocabulary, transform):
        """

        :param root: Directory where images are stored.
        :param trainingdatajson: JSON file which contains image names and captions
        :param vocabulary: Vocabulary for the captions data. For word this contains an index,
        which is specific to out dataset
        :param transform: Transform object that should be applied on each image.
        """
        self.root = root
        self.trainingdatajson = trainingdatajson
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, index):
        """
        This method returns an image and corresponding caption present at index: index
        :param index: index of the sample in self.trainingdatajson
        :return: image and caption pair
        """
        image_name, caption = self.trainingdatajson[index][0][0], self.trainingdatajson[index][0][1]
        image = Image.open(os.path.join(self.root, image_name)).convert('RGB')

        image = image.resize((256, 256), Image.ANTIALIAS)

        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())

        captions = []
        captions.append(self.vocabulary('<start>'))
        captions.extend([self.vocabulary(token) for token in tokens])
        captions.append(self.vocabulary('<end>'))
        target = torch.Tensor(captions)

        return image, target

    def __len__(self):
        return len(self.trainingdatajson)


def collect_batch_data(data):
    """
    This function creates a mini batch tensors from the list of tuples (image, caption).

    :param data: list of tuple(image, caption)
    :return: images, targets(batch * paddedlength), squ_lengths(list containing length of each caption seq)
    """
    data.sort(key= lambda x: len(x[1]), reverse=True)

    images, captions = zip(*data)

    images = torch.stack(images, 0)

    caption_lengths = [len(caption) for caption in captions]

    targets = torch.zeros(size=(len(data), max(caption_lengths))).long()

    for index, caption in enumerate(captions):
        targets[index][0:caption_lengths[index]] = caption[:caption_lengths[index]]

    return images, targets, caption_lengths


def get_loader(root, trainingdatajson, vocabulary, transform, batch_size, shuffle, num_workers):
    dataset = FlickrDataset(root, trainingdatajson, vocabulary, transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collect_batch_data)
    return data_loader



if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    with open("Data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open("Data/trainingdata.json") as json_file:
        trainingdatajson = json.load(json_file)
    data_loader = get_loader("Flicker8k_Dataset", trainingdatajson, vocab, transform, 128, True, 2)
    for i, (images, captions, lengths) in enumerate(data_loader):
        print("Jello")