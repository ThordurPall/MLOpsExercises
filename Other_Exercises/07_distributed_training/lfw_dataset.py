"""
LFW dataloading
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    """Labeled Faces in the Wild dataset"""

    def __init__(self, path_to_folder: str, transform) -> None:
        """
        Args:
            path_to_folder (string): Path to directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_to_folder = path_to_folder
        self.transform = transform

        # Find and store the labels and image paths
        self.labels, self.image_paths = [], []
        dirs = os.listdir(self.path_to_folder)

        for label in dirs:
            path = os.path.join(self.path_to_folder, label)
            images = os.listdir(path)

            for image in images:
                self.labels.append(label)
                self.image_paths.append(os.path.join(path, image))
        # self.image_paths = glob.glob(self.path_to_folder + '/*.jpg')
        self.name_to_label = {class_name: id for id,
                              class_name in enumerate(dirs)}

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index: int) -> torch.Tensor:
        'Generate a single sample from the dataset'

        # Get and transform the image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image_trans = self.transform(image)

        # Get the label and the class as an integer
        label = self.labels[index]
        label_id = self.name_to_label[label]
        return image_trans, label_id, label


def use_lfw_dataloading(path_to_folder, num_workers,
                        visualize_batch, get_timing):

    lfw_trans = transforms.Compose([
        # transforms.RandomAffine(5, (10, 10), (0.5, 2.0)),
        # transforms.RandomAffine(5, (1.0, 1.0), (0.5, 2.0)),
        transforms.RandomAffine(50),
        transforms.ToTensor()
    ])

    # Define dataset
    dataset = LFWDataset(path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=num_workers)

    if visualize_batch:
        # Visualize a batch of images
        images, label_ids, labels = next(iter(dataloader))
        plt.figure(figsize=(20, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            #plt.imshow(images[i][0])
            plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
            plt.title([labels[i] + ' (' +
                       str(label_ids[i].item()) + ')'])
            plt.axis('off')
        plt.show()

    if get_timing:
        # lets do so repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                # simulate that we do something with the batch
                #time.pause(0.2)
                #time.sleep(0.2)
                if batch_idx>100:
                    break

            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
        return np.mean(res), np.std(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw', type=str) # default='', type=str)
    parser.add_argument('-num_workers', default=0, type=int) # default=None, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-make_errorbar_plot', action='store_true')
    args = parser.parse_args()

    use_lfw_dataloading(args.path_to_folder, args.num_workers,
                        args.visualize_batch, args.get_timing)

    if args.make_errorbar_plot:
        means, stds = [], []
        for i in range(4):
            t_mean, t_std = use_lfw_dataloading('lfw', i+1,
                                                False, True)
            means.append(t_mean)
            stds.append(t_std)
        print(means)
        print(stds)
        f = plt.figure(figsize=(12, 8))
        plt.errorbar(list(range(1, 5)), means, yerr=stds)
        plt.xlabel('Number of workers')
        plt.ylabel('Time')
        plt.plot()
        f.savefig('./errorbar_plot.pdf', bbox_inches='tight')
