"""
This file reads the names of training images and stores them and corresponding captions
in an array and stores this a s json.
"""
import json
import argparse


def create_data_json(captions_file, images_name, json_file_path):
    """

    :param captions_file: A file which contains image name and its corresponding caption
    :param images_name: A file which contains list of names of training images.
    :param json_file_path: Location to the output JSON file.
    :return: stores in /Data folder.
    """
    images = open(images_name, "r")

    images = [line.strip() for line in list(images)]

    training_data = {}

    for caption_label in open(captions_file, "r"):
        image_name, caption = caption_label.split(maxsplit=1)
        image_name = image_name.split("#")[0]
        # Only considering the images in data.
        if image_name in images:
            caption = caption.strip()
            if image_name not in training_data:
                training_data[image_name] = ([image_name], [])
            training_data[image_name][1].append(caption)

    with open(json_file_path, "w") as output_json_file:
        json.dump(list(training_data.values()), output_json_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainimages_path', type=str,
                        default='/Users/demo/Downloads/santosh/imagecaptioning/Flickr8k_text/Flickr_8k.devImages.txt',
                        help='Path for list of images')
    parser.add_argument('--caption_path', type=str,
                        default='/Users/demo/Downloads/santosh/imagecaptioning/Flickr8k_text/Flickr8k.lemma.token.txt',
                        help='Path for captions')
    parser.add_argument('--trainingdata_json_path', type=str, default='/Users/demo/Downloads/santosh/imagecaptioning/Data/devdata.json',
                        help='path for data json')

    args = parser.parse_args()
    create_data_json(args.caption_path, args.trainimages_path, args.trainingdata_json_path)



