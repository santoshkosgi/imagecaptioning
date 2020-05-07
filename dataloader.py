"""
This function has methods to load the images and their captions.
Preprocess the images to the format needed by resnet 101 and split the data to train, test and
validation.
"""

import torchvision.datasets as dset
import torch.utils.data as data


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