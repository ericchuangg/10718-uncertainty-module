import argparse
import random
import numpy as np
import os
import glob

import pandas as pd
import seaborn
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate CNN")

    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=[1, 2, 4, 7, 10])

    parser.add_argument("--inputs", type=str, default="*/results.csv")

    parser.add_argument("--out", type=str, default="accuracy-vs-threshold.png")

    args = parser.parse_args()

    df = pd.concat([
        pd.read_csv(file, index_col=0) 
        for file in glob.glob(args.inputs)
    ], ignore_index=True)

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    all_datasets = list(df["Dataset"].unique())
    all_ensemble_sizes = list(df["Ensemble Size"].unique())

    fig, axs = plt.subplots(1, len(all_datasets), figsize=(6 * len(all_datasets), 6))

    color_palette = seaborn.color_palette(n_colors=len(all_ensemble_sizes))

    for i, dataset in enumerate(all_datasets):

        selected_df = df[df["Dataset"] == dataset]

        axis = seaborn.lineplot(data=selected_df, 
                                x="Confidence Threshold", 
                                y="Validation Accuracy", 
                                hue="Ensemble Size", 
                                linewidth=4, ax=axs[i], 
                                palette=color_palette, 
                                errorbar=('ci', 68))

        if i == 0: handles, labels = axis.get_legend_handles_labels()
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

        axis.set_title(f"Dataset = {dataset}",
                        fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(handles, all_ensemble_sizes,
                        loc="lower center", ncol=len(all_ensemble_sizes),
                        prop={'size': 24, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(color_palette[i])

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig(args.out)
