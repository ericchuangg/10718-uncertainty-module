import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import models
from torchvision import transforms

import argparse
import random
import numpy as np
import os
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


DATASETS = {
    "mnist": datasets.MNIST,
    "cifar": datasets.CIFAR10,
    "caltech": datasets.Caltech101,
    "eurosat": datasets.EuroSAT,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train CNN")

    parser.add_argument("--dataset", type=str, default="mnist", choices=DATASETS.keys())

    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--ensemble-size", type=int, default=10)

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--out", type=str, default="mnist-ensemble")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    image_size = (args.image_size, args.image_size)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Lambda(lambda x: x.expand(3, *image_size)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    try:  # attempt to use the official train-val split

        dataset_train = DATASETS[args.dataset](
            ".", train=True, transform=transform, download=True)
        dataset_val = DATASETS[args.dataset](
            ".", train=False, transform=transform, download=True)

    except TypeError:  # manually create one if not available

        dataset_train = DATASETS[args.dataset](
            ".", transform=transform, download=True)

        dataset_train, dataset_val = \
            torch.utils.data.random_split(dataset_train, [0.8, 0.2])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True)

    for ensemble_idx in range(args.ensemble_size):

        model = models.resnet18(num_classes=args.num_classes).cuda()

        optim = torch.optim.Adam(model.parameters(), lr=0.0003)

        for epoch in range(args.epochs):

            model.train()

            training_accuracy = []

            for iteration, (images, labels) in enumerate(dataloader_train):

                images = images.cuda()
                labels = labels.cuda()

                predictions = model(images)

                loss = F.cross_entropy(predictions, labels)

                loss = loss.mean()
                loss.backward()

                optim.step()
                optim.zero_grad()

                training_accuracy.append(
                    predictions.argmax(dim=-1) == labels
                )

                if (iteration + 1) % 50 == 0:

                    print("[{}] Ensemble {:02d}  Iteration {:05d}  Training Accuracy: {:0.5f}".format(
                        args.dataset, ensemble_idx, iteration, torch.cat(
                            training_accuracy, dim=0
                        ).float().mean().cpu().numpy().item()))

            model.eval()

            validation_accuracy = []

            for iteration, (images, labels) in enumerate(dataset_val):

                images = images.cuda()
                labels = labels.cuda()

                predictions = model(images)

                validation_accuracy.append(
                    predictions.argmax(dim=-1) == labels
                )

            print("[{}] Ensemble {:02d}  Epoch {:05d}  Validation Accuracy: {:0.5f}".format(
                args.dataset, ensemble_idx, epoch, torch.cat(
                    validation_accuracy, dim=0
                ).float().mean().cpu().numpy().item()))

        torch.save(model.state_dict(), os.path.join(
            args.out, "model-{:02d}.pt".format(ensemble_idx)))
