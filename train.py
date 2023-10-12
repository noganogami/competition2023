import argparse
import csv
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n_fold", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="efficientnetv2_rw_s")
    parser.add_argument("--car", type=str, default="4runner")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--path", type=str, default="model_wts.pth")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seed", type=int, default=114514)

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, img_pathes, shanai_pathes, transform):
        self.img_pathes = img_pathes
        self.shanai_pathes = shanai_pathes
        self.transform = transform
        self.imgs = [self.__getitem__(i)[0] for i in range(len(img_pathes))]
        self.targets = [self.__getitem__(i)[1] for i in range(len(img_pathes))]
        self.classes = ["soto", "naka"]

    def __getitem__(self, idx):
        img_path = self.img_pathes[idx]
        label = 1 if os.path.split(img_path)[1] in self.shanai_pathes else 0
        img = self.image_loader(img_path)
        if self.transform:
            img = transform(img)

        return img, label

    def __len__(self):
        return len(self.img_pathes)

    def image_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.imgs = [dataset.imgs[i] for i in indices]
        self.targets = [dataset.targets[i] for i in indices]
        self.classes = dataset.classes

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = (
            loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma
        )  # focal loss

        return loss.sum()


class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes, head="linear", feat_dim=None):
        super(CustomModel, self).__init__()
        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=0,
        )
        self.num_classes = num_classes
        self.in_features = self.encoder.num_features
        feat_dim = self.in_features if feat_dim is None else feat_dim

        if head == "linear":
            self.head = nn.Linear(self.in_features, num_classes)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.in_features, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, num_classes),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

    def forward(self, images):
        features = self.encoder(images)
        logits = self.head(features)
        return logits


class Metrics:
    def __init__(self):
        self.tng_loss = []
        self.tng_acc = []
        self.val_loss = []
        self.val_acc = []

    def save_learning_curves(self):
        plt.figure()

        plt.plot(self.tng_loss, label="train")
        plt.plot(self.val_loss, label="validation")

        plt.xlabel("Number of samples")
        plt.ylabel("Loss")
        plt.legend()

        path = "loss.png"
        plt.savefig("loss.png")
        print(f"loss curve is saved as {path}")

        plt.figure()

        plt.plot(self.tng_acc, label="train")
        plt.plot(self.val_acc, label="validation")

        plt.xlabel("Number of samples")
        plt.ylabel("acc")
        plt.legend()

        path = "acc.png"
        plt.savefig("acc.png")
        print(f"acc curve is saved as {path}")


def get_input_size(model_name):
    c, h, w = timm.get_pretrained_cfg_value(model_name, "input_size")
    return h, w


def build_dataset(shanai_df, transforms=None, car="4runner"):
    pathes = glob(f"toyota_cars/{car}/*")
    shanais = list(shanai_df["file_name"])
    dataset = CustomDataset(
        img_pathes=pathes, shanai_pathes=shanais, transform=transforms
    )
    return dataset


def build_transforms(height, width):
    tng_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(height, width), scale=(0.5, 1.0), ratio=(1.0, 1.0)
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(size=max(height, width)),
            transforms.CenterCrop(size=(height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tng_transforms, val_transforms


def train(model, optimizer, criterion, dataloader, scheduler, device, tracker):
    model.train()
    running_loss = 0
    running_acc = 0
    size = len(dataloader.dataset)

    with tqdm(dataloader, desc="[Train]") as pbar:
        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)

            logits = model(img)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            _, preds = torch.max(logits, 1)

            running_loss += loss.detach().item()
            running_acc += torch.sum(preds == label.detach()).item()
        tracker.tng_loss.append(running_loss / size)
        tracker.tng_acc.append(running_acc / size)


def validation(model, criterion, dataloader, device, tracker):
    model.eval()
    running_loss = 0
    running_acc = 0
    size = len(dataloader.dataset)
    pred_list = []

    with tqdm(dataloader, desc="[Valid]") as pbar:
        with torch.no_grad():
            for img, label in pbar:
                img = img.to(device)
                label = label.to(device)

                logits = model(img)
                loss = criterion(logits, label)

                _, preds = torch.max(logits, 1)

                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == label.detach()).item()
                pred_list.append(preds.cpu())

        tracker.val_loss.append(running_loss / size)
        tracker.val_acc.append(running_acc / size)


def prediction(model, dataset, device, transform, car):
    pred_list = []

    with torch.no_grad():
        with open(f"{car}.csv", "a", newline="") as f:
            with open(f"dif.csv", "a", newline="") as d:
                writer = csv.writer(f)
                d_witer = csv.writer(d)

                for idx, (img, _) in enumerate(tqdm(dataset)):
                    img = transform(img)
                    img = img.unsqueeze(0).to(device)
                    logit = model(img)
                    prob = F.softmax(logit, dim=1)

                    pred_list.append(logit.argmax().item())
                    if prob[0][1] >= 0.7:
                        writer.writerow([os.path.split(dataset.img_pathes[idx])[1]])
                    elif 0.3 < prob[0][1] and prob[0][1] < 0.7:
                        d_witer.writerow([dataset.img_pathes[idx], prob[0][1].item()])
    print(
        classification_report(
            y_true=dataset.targets,
            y_pred=pred_list,
            target_names=dataset.classes,
            digits=3,
        )
    )


def train_cross_validation(
    model,
    dataset,
    tng_transform,
    val_transform,
    learning_rate,
    batch_size,
    n_epochs,
    n_fold,
    device,
    seed,
    scheduler=None,
):
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(params=model.head.parameters(), lr=learning_rate)

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = FocalLoss(gamma=1)

    tracker = Metrics()

    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for fold, (tng_idx, val_idx) in enumerate(skf.split(dataset.imgs, dataset.targets)):
        print(f"Fold {fold+1}/{n_fold}")
        print("-" * 10)

        tng_dataset = CustomSubset(dataset, tng_idx, tng_transform)
        val_dataset = CustomSubset(dataset, val_idx, val_transform)

        tng_dataloader = DataLoader(tng_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        with tqdm(range(n_epochs)) as pbar:
            for i in pbar:
                pbar.set_description(f"[Epoch {i+1}]")

                train(
                    model=model,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    dataloader=tng_dataloader,
                    scheduler=scheduler,
                    device=device,
                    tracker=tracker,
                )
                validation(
                    model=model,
                    dataloader=val_dataloader,
                    criterion=loss_fn,
                    device=device,
                    tracker=tracker,
                )

                logs = {
                    "tng_loss": "%.3f" % tracker.tng_loss[i],
                    "tng_acc": "%.3f" % tracker.tng_acc[i],
                    "val_loss": "%.3f" % tracker.val_loss[i],
                    "val_acc": "%.3f" % tracker.val_acc[i],
                }
                pbar.set_postfix(**logs)

    tracker.save_learning_curves()


def main():
    args = parse_option()

    df = pd.read_csv("shanai.csv")
    dataset = build_dataset(df, car=args.car)
    h, w = get_input_size(args.model_name)
    tng_trans, val_trans = build_transforms(h, w)

    model = CustomModel(
        model_name=args.model_name, num_classes=len(dataset.classes), head="mlp"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.train:
        train_cross_validation(
            model=model,
            dataset=dataset,
            tng_transform=tng_trans,
            val_transform=val_trans,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            n_fold=args.n_fold,
            device=device,
            seed=args.seed,
            scheduler=None,
        )
        torch.save(model.state_dict(), args.path)
        print(f"Model is saved as '{args.path}'")
    else:
        model.load_state_dict(torch.load(args.path))
        print(f"Model is loaded from '{args.path}'")

    prediction(
        model=model, dataset=dataset, device=device, transform=val_trans, car=args.car
    )


if __name__ == "__main__":
    main()
