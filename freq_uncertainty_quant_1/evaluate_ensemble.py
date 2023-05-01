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

import pandas as pd
import seaborn
import matplotlib
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context


DATASETS = {
    "mnist": datasets.MNIST,
    "cifar": datasets.CIFAR10,
    "caltech": datasets.Caltech101,
    "eurosat": datasets.EuroSAT,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate CNN")

    parser.add_argument("--dataset", type=str, default="mnist", choices=DATASETS.keys())

    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=[1, 2, 4, 7, 10])

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--out", type=str, default="mnist-ensemble")

    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])

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
        dataset_train, batch_size=args.batch_size, shuffle=False)

    dataset_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False)

    ensemble_logits = []
    ensemble_labels = None

    for ensemble_idx in range(max(args.ensemble_sizes)):

        model = models.resnet18(num_classes=args.num_classes).cuda()

        model.load_state_dict(torch.load(os.path.join(
            args.out, "model-{:02d}.pt".format(ensemble_idx))))

        model.eval()

        validation_logits = []
        validation_labels = []

        for iteration, (images, labels) in enumerate(dataset_val):

            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                predictions = model(images)

            validation_logits.append(
                predictions
            )

            if ensemble_labels is None:
                validation_labels.append(labels)

        ensemble_logits.append(
            torch.cat(validation_logits, dim=0)
        )

        if ensemble_labels is None:
            ensemble_labels = torch.cat(validation_labels, dim=0)
        
    ensemble_logits = torch.stack(ensemble_logits, dim=0)

    all_records = []

    for ensemble_size in args.ensemble_sizes:

        ensemble_logits_i = ensemble_logits[:ensemble_size].mean(dim=0)

        probs, indices = torch.softmax(
            ensemble_logits_i, dim=-1
        ).max(dim=-1)

        accuracy = (indices == ensemble_labels).float()

        print("[{}]  Validation Accuracy: {:0.5f}".format(
            args.dataset, accuracy.mean().cpu().numpy().item()))

        for threshold in args.thresholds:

            threshold_accuracy = accuracy[torch.argwhere(probs > threshold)]

            all_records.append({
                "Dataset": args.dataset,
                "Ensemble Size": ensemble_size,
                "Confidence Threshold": threshold,
                "Validation Accuracy": threshold_accuracy.mean().cpu().numpy().item(),
            })

    df = pd.DataFrame.from_records(all_records)

    df.to_csv(os.path.join(args.out, "results.csv"))

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    color_palette = seaborn.color_palette(n_colors=len(args.ensemble_sizes))

    axis = seaborn.lineplot(data=df, x="Confidence Threshold", 
                            y="Validation Accuracy", 
                            hue="Ensemble Size", 
                            linewidth=4, ax=axs, 
                            palette=color_palette, 
                            errorbar=('ci', 68))

    handles, labels = axis.get_legend_handles_labels()
    axis.legend([],[], frameon=False)

    axis.set(xlabel=None)
    axis.set(ylabel=None)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16, labelrotation=45)

    axis.set_xlabel("Confidence Threshold ($\\tau$)", fontsize=24,
                    fontweight='bold', labelpad=12)

    axis.set_ylabel("Accuracy (Val)", fontsize=24,
                    fontweight='bold', labelpad=12)

    axis.set_title(f"Dataset = {args.dataset}",
                    fontsize=24, fontweight='bold', pad=12)

    axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(handles, args.ensemble_sizes,
                        loc="lower center", ncol=len(args.ensemble_sizes),
                        prop={'size': 24, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(color_palette[i])

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig(os.path.join(args.out, "accuracy-vs-threshold.png"))
