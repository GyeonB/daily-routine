import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json
import numpy as np
import pandas as pd
from collections import Counter
from pprint import pprint
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import time
import copy
from tqdm import tqdm

BATCH_SIZE = 128
LEARNING_RATE = 1e-8
MOMENTUM = 0.9
NUM_EPOCHS = 50
LOG_INTERVAL = 500
CHECKPOINT_DIR = "./checkpoints/"
EARLY_STOPPING_PATIENCE = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

logging.basicConfig(filename='training_augmentation.log', level=logging.INFO, format='%(asctime)s %(message)s')

def create_dataloader(dataset_dir, transform, batch_size):
    if '.ipynb_checkpoints' in os.listdir(dataset_dir):
        os.rmdir(dataset_dir + '.ipynb_checkpoints/')
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataset, dataloader

train_dataset, train_dataloader = create_dataloader('../datasets/train/', train_transform, BATCH_SIZE)
test_dataset, test_dataloader = create_dataloader('../datasets/test/', val_test_transform, BATCH_SIZE)
val_dataset, val_dataloader = create_dataloader('../datasets/val/', val_test_transform, BATCH_SIZE)

train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None, num_classes=len(train_dataset.classes))
model = model.to(device)

cost_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

def compute_class_weights(dataset):
    labels = [label for _, label in dataset]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth.tar'))

def train_model(model, cost_function, datasets: list, optimizer, num_epochs=25, patience=10, checkpoint_dir='./checkpoints'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    class_weights = compute_class_weights(datasets[0].dataset)
    cost_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(num_epochs):
        train_size = 0
        val_size = 0
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_loader = tqdm(datasets[0], desc=f"Epoch {epoch}/{num_epochs - 1} Training Batches", leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            train_size += inputs.size(0)
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = cost_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % LOG_INTERVAL == 0:
                logging.info(f'Batch {batch_idx}/{len(datasets[0])} - Loss: {loss.item():.4f}')
                
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu().numpy())

        logging.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        val_loader = tqdm(datasets[1], desc=f"Epoch {epoch}/{num_epochs - 1} Validation Batches", leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                val_size += inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = cost_function(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if batch_idx % LOG_INTERVAL == 0:
                    logging.info(f'Batch {batch_idx}/{len(datasets[1])} - Val Loss: {loss.item():.4f}')

        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size

        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc.cpu().numpy())

        logging.info(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0

            # Save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
            }, is_best=True, checkpoint_dir=checkpoint_dir)
        else:
            epochs_no_improve += 1

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot_aug.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(train_accuracies, label="train")
    plt.plot(val_accuracies, label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot_aug.png")
    plt.close()

    return model

model = train_model(model, cost_function, [train_dataloader, val_dataloader], optimizer, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE, checkpoint_dir=CHECKPOINT_DIR)

torch.save(model, 'ResNet_Augmentation.pt')

model.eval()
running_corrects = 0

test_loader = tqdm(test_dataloader, desc="Testing Batches", leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % LOG_INTERVAL == 0:
            logging.info(f'Test Batch {batch_idx}/{len(test_dataloader)}')

test_acc = running_corrects.double() / len(test_dataset)
logging.info(f'Test Acc: {test_acc:.4f}')
