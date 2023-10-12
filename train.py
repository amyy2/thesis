import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from load_dataset import BraTSDataset
from model import UNet3D
import torch.nn.functional as F

device = torch.device("cuda")
learning_rate = 0.0001
batch_size = 1
num_epochs = 100

def f1(probability, targets):
    intersection = 2.0 * (probability * targets).sum()
    union = (probability * probability).sum() + (targets * targets).sum()
    dice_score = intersection / union
    return 1.0 - dice_score

def f1_metric(model, loader):
    f1_score = 0.0
    with torch.no_grad():
        for data, seg in loader:

            data = data.cuda()
            seg = seg.cuda()
            data = data.unsqueeze(1)
            pred = model(data.float())

            output_softmax = F.softmax(pred, dim=1)

            mapping = {0: 0, 1: 1, 2: 2, 4: 3}
            for old_label, new_label in mapping.items():
                seg[seg == old_label] = new_label

            f1_score += f1(output_softmax, F.one_hot(seg.long(), num_classes=4).permute(0, 4, 1, 2, 3))

    f1_score /= len(loader)

    return f1_score.item()

def accuracy(model, loader, prin=False):
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for data, seg in loader:
            data = data.cuda()
            seg = seg.cuda()
            data = data.unsqueeze(1)
            pred = model(data.float())
            pred = torch.argmax(pred, dim=1)

            correct_pixels += (pred == seg).sum()
            total_pixels += torch.numel(pred)

    return (correct_pixels / total_pixels).item()

model = UNet3D(in_channels=1, out_channels=4).to(device)
model = model.float()

loss_function = f1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = BraTSDataset('archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')

validation_dataset = BraTSDataset('archive/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData')

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

def train():

    scores = {'accuracy': [], 'f1': []}

    for epoch in range(num_epochs):

        scores["accuracy"].append(accuracy(model, validation_loader))
        scores["f1"].append(f1_metric(model, validation_loader))

        model.train()
        progress = tqdm(train_loader)

        for batch, (image, mask) in enumerate(progress):

            image = image.cuda()
            image = image.unsqueeze(1)
            mask = mask.cuda()

            output = model(image.float())
            output_softmax = F.softmax(output, dim=1)

            mapping = {0: 0, 1: 1, 2: 2, 4: 3}
            for old_label, new_label in mapping.items():
                mask[mask == old_label] = new_label

            target_one_hot = F.one_hot(mask.long().squeeze(0), num_classes=4).permute(3, 0, 1, 2).float()

            loss = loss_function(output_softmax, target_one_hot.unsqueeze(1).permute(1, 0, 2, 3, 4))

            progress.set_postfix(loss=loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()

        scores["accuracy"].append(accuracy(model, validation_loader))
        scores["f1"].append(f1_metric(model, validation_loader))

        if epoch % 10 == 0:
            with torch.no_grad():
                image_cpu = image.squeeze().cpu().numpy()
                mask_cpu = mask.squeeze().cpu().numpy()
                output_cpu = output_softmax.squeeze().argmax(dim=0).cpu().numpy()

                slice_idx = image_cpu.shape[0] // 2

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(image_cpu[slice_idx], cmap='gray')
                axs[0].set_title('Input')
                axs[1].imshow(output_cpu[slice_idx], cmap='gray')
                axs[1].set_title('Output')
                axs[2].imshow(mask_cpu[slice_idx], cmap='gray')
                axs[2].set_title('Target')
                plt.show()

        print("Accuracy for epoch (" + str(epoch) + ") is: " + str(scores["accuracy"][-1]))
        print("F1 Score for epoch (" + str(epoch) + ") is: " + str(scores["f1"][-1]))

        plt.figure(figsize=(10, 10), dpi=100)
        plt.scatter(range(0, epoch + 1), scores["accuracy"])
        plt.title("accuracy")
        plt.show()
        plt.figure(figsize=(10, 10), dpi=100)
        plt.scatter(range(0, epoch + 1), scores["f1"])
        plt.title("f1")
        plt.show()

    print("Final Accuracy for epoch is: " + str(accuracy(model, validation_loader)))
    print("Final DSC Score for epoch is: " + str(f1_metric(model, validation_loader)))

if __name__ == "__main__":
    train()