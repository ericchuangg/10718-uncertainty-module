import pandas as pd
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import models
from torchvision import transforms

import pandas as pd
import seaborn
import matplotlib
import matplotlib.pyplot as plt


class PandasDataset(torch.utils.data.Dataset):
 
    def __init__(self, features, labels):

        self.features = features
        self.labels = labels.to_numpy()

        self.keys = self.features.columns

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self, idx):

        row = self.features.iloc[idx].to_dict()

        features = torch.tensor([
            float(row[key]) for key in self.keys
        ], dtype=torch.float32)

        return features, self.labels[idx]


def experiment(i, train_df, train_labels, val_df, val_labels):

    dataset_train = PandasDataset(train_df, train_labels)
    dataset_val = PandasDataset(val_df, val_labels)

    features_mean = torch.stack([
        x[0] for x in dataset_train
    ], dim=0).cuda().mean(dim=0, keepdim=True)

    features_std = torch.stack([
        x[0] for x in dataset_train
    ], dim=0).cuda().std(dim=0, keepdim=True)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True)

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=32, shuffle=False)

    ensemble = []

    for ensemble_idx in range(10):

        model = nn.Sequential(
            nn.Linear(dataset_train[0][0].shape[0], 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1000)
        ).cuda()

        optim = torch.optim.Adam(model.parameters(), lr=0.0003)

        for epoch in range(1):

            model.train()

            training_accuracy = []

            for iteration, (features, labels) in enumerate(dataloader_train):

                features = features.cuda()
                labels = labels.cuda()

                features = (features - features_mean) / (features_std + 1e-6)

                predictions = model(features)

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
                        "donors-choose", ensemble_idx, iteration, torch.cat(
                            training_accuracy, dim=0
                        ).float().mean().cpu().numpy().item()))

            model.eval()

            validation_accuracy = []

            for iteration, (features, labels) in enumerate(dataloader_val):

                features = features.cuda()
                labels = labels.cuda()

                features = (features - features_mean) / (features_std + 1e-6)

                predictions = model(features)

                validation_accuracy.append(
                    predictions.argmax(dim=-1) == labels
                )

            print("[{}] Ensemble {:02d}  Epoch {:05d}  Validation Accuracy: {:0.5f}".format(
                "donors-choose", ensemble_idx, epoch, torch.cat(
                    validation_accuracy, dim=0
                ).float().mean().cpu().numpy().item()))

        torch.save(model.state_dict(), os.path.join(
            "donors-choose-ensembles", 
            "model-{:01d}-{:02d}.pt".format(i, ensemble_idx)))

    ensemble_logits = []
    ensemble_labels = None

    ensemble_sizes = [1, 2, 4, 7, 10]

    for ensemble_idx in range(max(ensemble_sizes)):

        model.load_state_dict(torch.load(os.path.join(
            "donors-choose-ensembles", 
            "model-{:01d}-{:02d}.pt".format(i, ensemble_idx))))

        model.eval()

        validation_logits = []
        validation_labels = []

        for iteration, (features, labels) in enumerate(dataloader_val):

            features = features.cuda()
            labels = labels.cuda()

            features = (features - features_mean) / (features_std + 1e-6)

            with torch.no_grad():
                predictions = model(features)

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

    for ensemble_size in ensemble_sizes:

        ensemble_logits_i = ensemble_logits[:ensemble_size]

        probs, indices = torch.softmax(
            ensemble_logits_i, dim=-1
        ).mean(dim=0).max(dim=-1)

        accuracy = (indices == ensemble_labels).float()

        print("[{}]  Validation Accuracy: {:0.5f}".format(
            "Donors Choose", accuracy.mean().cpu().numpy().item()))

        for threshold in [0.0, 0.2, 0.4, 0.6, 0.8]:

            threshold_accuracy = accuracy[torch.argwhere(probs > threshold)]

            all_records.append({
                "Dataset": f"Donors Choose (Split = {i})",
                "Ensemble Size": ensemble_size,
                "Confidence Threshold": threshold,
                "Validation Accuracy": threshold_accuracy.mean().cpu().numpy().item(),
            })

    df = pd.DataFrame.from_records(all_records)

    df.to_csv(os.path.join("donors-choose-ensembles", f"results-{i}.csv"))

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    color_palette = seaborn.color_palette(n_colors=len(ensemble_sizes))

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

    axis.set_title(f"Dataset = Donors Choose",
                    fontsize=24, fontweight='bold', pad=12)

    axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(handles, ensemble_sizes,
                        loc="lower center", ncol=len(ensemble_sizes),
                        prop={'size': 24, 'weight': 'bold'})

    for j, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(color_palette[j])

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig(os.path.join(
        "donors-choose-ensembles", 
        f"accuracy-vs-threshold-{i}.png"))

    return train_df, val_df


if __name__ == "__main__":

    os.makedirs("donors-choose-ensembles", exist_ok=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    donations = pd.read_csv('donors-choose-data/donations.csv')
    essays = pd.read_csv('donors-choose-data/essays.csv')
    outcomes = pd.read_csv('donors-choose-data/outcomes.csv')
    projects = pd.read_csv('donors-choose-data/projects.csv')
    resources = pd.read_csv('donors-choose-data/resources.csv')

    projects_donations = projects.merge(donations[[
        "donationid", "projectid", "donation_timestamp", 
        "donation_to_project"]], on="projectid")

    new_donations = donations.merge(projects[["date_posted", "projectid"]], on="projectid", how="left")

    new_donations["donation_to_project"] = new_donations["donation_to_project"] * (
        (pd.to_datetime(new_donations["donation_timestamp"], format='mixed', dayfirst=False) - 
         pd.to_datetime(new_donations["date_posted"])) < pd.Timedelta("120day")).astype(float)

    joined = projects.join(new_donations.groupby("projectid")[
        "donation_to_project"].sum(), on="projectid", how="left")

    joined = joined.fillna(0)

    joined["fraction_funded"] = (
        joined["donation_to_project"] / 
        joined["total_price_excluding_optional_support"])

    joined["fully_funded"] = 0
    joined.loc[joined["fraction_funded"] >= 1, "fully_funded"] = 1

    joined["date_posted"] = pd.to_datetime(joined["date_posted"])

    codes, uniques = pd.factorize(joined["poverty_level"])
    print(uniques)
    joined["poverty_level_num"] = codes.astype(float)

    high_priority_features = [
        "teacher_teach_for_america", "teacher_ny_teaching_fellow", "primary_focus_subject", 
        "primary_focus_area", "resource_type", "poverty_level", 
        "total_price_excluding_optional_support", "students_reached"]

    starting_date = pd.to_datetime("2008/01/01")

    training_set_length = pd.Timedelta("360days")
    validation_set_length = pd.Timedelta("120days")

    model_retrain_interval = pd.Timedelta("360days")

    records_df = []

    high_priority_df = joined[high_priority_features]
    high_priority_df = pd.get_dummies(high_priority_df, columns = [
        "teacher_teach_for_america", "teacher_ny_teaching_fellow", 
        "primary_focus_subject", "primary_focus_area", "resource_type", "poverty_level"])

    for i in range(3):  # iterate through the invervals

        training_set_start = starting_date + model_retrain_interval * i
        training_set_end = training_set_start + training_set_length
        validation_set_end = training_set_end + validation_set_length

        training_set = high_priority_df[(joined['date_posted'] >= training_set_start) & (
            joined['date_posted'] < training_set_end)]

        validation_set = high_priority_df[(joined['date_posted'] >= training_set_end) & (
            joined['date_posted'] < validation_set_end)]

        print("training_set_start:", training_set_start,  
            "training_set_end:", training_set_end, 
            "start_date_for_labels:", training_set_start + pd.Timedelta("120days"), 
            "end_date_for_labels:", training_set_end + pd.Timedelta("120days"))

        print("validation_set_start:", training_set_end,  
            "validation_set_end:", validation_set_end, 
            "start_date_for_labels:", training_set_end + pd.Timedelta("120days"), 
            "end_date_for_labels:", validation_set_end + pd.Timedelta("120days"))
        
        training_labels = joined[(joined['date_posted'] >= training_set_start) & (
            joined['date_posted'] < training_set_end)]['fully_funded']
        validation_labels = joined[(joined['date_posted'] >= training_set_end) & (
            joined['date_posted'] < validation_set_end)]['fully_funded'] 

        training_set, validation_set = experiment(
            i, training_set, training_labels, 
            validation_set, validation_labels)
