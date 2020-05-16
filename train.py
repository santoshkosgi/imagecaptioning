import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_dir, "..", "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_dir, "..", ".."))

from dataloader import get_loader
from build_vocabulary import Vocabulary
from models import Encoder
from models import Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import json

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.caption_path) as json_file:
        trainingdatajson = json.load(json_file)

    # Build data loader
    data_loader = get_loader(args.image_dir, trainingdatajson, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = Encoder(args.embed_size)

    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    # Load saved model
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))


    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)

    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images
            captions = captions

            # Targets are like this because, we add embedding layer input at the start of every seq in batch
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, backward and optimize
            features = encoder(images)
            output = decoder(features, captions, lengths)
            loss = criterion(output, targets)
            loss.backward()
            encoder.zero_grad()
            decoder.zero_grad()
            optimizer.step()

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--encoder_path', type=str, default='models/encoder_old.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder_old.ckpt',
                        help='path for trained decoder')
    # parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='Data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='Flicker8k_Dataset', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='Data/trainingdata.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
