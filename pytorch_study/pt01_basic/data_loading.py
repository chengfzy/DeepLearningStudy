from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from optparse import OptionParser
import util


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples"""
    images, landmarks = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images)
    img_size = images.size(2)

    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    for i in range(batch_size):
        plt.scatter(landmarks[i, :, 0].numpy() + i * img_size, landmarks[i, :, 1].numpy(), s=10, marker='.', c='r')
        plt.title('Batch from dataloader')


class FaceLandmarksDataset(Dataset):
    """Face landmarks dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Face landmark dataset
        :param csv_file: Path to the csv file with annotations
        :param root_dir: Directory with all the images
        :param transform: Optional transform to be applied on a sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, "landmarks": landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because numpy image: HxWxC => torch image: CxHxzW
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


if __name__ == '__main__':
    # define some parameters
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', default='../../../../dataset/faces/', help='data set folder')
    options, args = parser.parse_args()

    # 1. simple usage
    print(util.Section('Simple Usage'))
    landmarks_frame = pd.read_csv(options.folder + 'face_landmarks.csv')
    n = 65
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].values
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    plt.figure()
    show_landmarks(io.imread(options.folder + img_name), landmarks)
    # plt.show()

    # 2. use as a Dataset
    print(util.Section('Use as a Dataset'))
    face_dataset = FaceLandmarksDataset(csv_file=os.path.join(options.folder, 'face_landmarks.csv'),
                                        root_dir=options.folder)
    fig = plt.figure()
    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)
        if i == 3:
            # plt.show()
            break

    # 3. resize, crop and transform
    print(util.Section('Resize, Crop and Transform'))
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    # apply each of the above transforms on sample
    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)
    # plt.show()

    # 4. put all together to create a dataset with composed transforms
    print(util.Section('Put All together to Create a Dataset'))
    transformed_dataset = FaceLandmarksDataset(csv_file=os.path.join(options.folder, 'face_landmarks.csv'),
                                               root_dir=options.folder,
                                               transform=transforms.Compose([Rescale(256),
                                                                             RandomCrop(224),
                                                                             ToTensor()]))
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['landmarks'].size())
        if i == 3:
            break

    # 5. Batch version
    print(util.Section('Batch Version'))
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batched, sample_batched in enumerate(dataloader):
        print(i_batched, sample_batched['image'].size(), sample_batched['landmarks'].size())

        # observe 4th batch and stop
        if i_batched == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            # plt.show()
            break

    plt.show()
