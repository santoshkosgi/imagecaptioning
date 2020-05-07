"""
This file reads the names of training images and stores them and corresponding captions
in an array and stores this a s json.
"""
import json
import argparse


def create_data_json(captions_file, training_images, json_file_path):
    """

    :param captions_file: A file which contains image name and its corresponding caption
    :param training_images: A file which contains list of names of training images.
    :param json_file_path: Location to the output JSON file.
    :return: stores in /Data folder.
    """
    train_images = open(training_images, "r")

    train_images = [line.strip() for line in list(train_images)]

    training_data = []

    for caption_label in open(captions_file, "r"):
        image_name, caption = caption_label.split(maxsplit=1)
        image_name = image_name.split("#")[0]
        # Only considering the images in training.
        if image_name in train_images:
            caption = caption.strip()
            training_data.append([(image_name, caption)])

    with open(json_file_path, "w") as output_json_file:
        json.dump(training_data, output_json_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainimages_path', type=str,
                        default='/Users/demo/Downloads/santosh/imagecaptioning/Flickr8k_text/Flickr_8k.trainImages.txt',
                        help='path for Images Annotation file')
    parser.add_argument('--caption_path', type=str,
                        default='/Users/demo/Downloads/santosh/imagecaptioning/Flickr8k_text/Flickr8k.lemma.token.txt',
                        help='path for Images Annotation file')
    parser.add_argument('--trainingdata_json_path', type=str, default='/Users/demo/Downloads/santosh/imagecaptioning/Data/trainingdata.json',
                        help='path for saving vocabulary wrapper')

    args = parser.parse_args()
    create_data_json(args.caption_path, args.trainimages_path, args.trainingdata_json_path)



