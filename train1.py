import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Constants
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Pleural_Thickening', 'No Finding'
]
DATA_DIR = r'D:/university/LEVEL 2/Semster 2/project/code'
CSV_FILE = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
BEST_MODEL_PATH = 'best_model_multi_label_nih_original_resnet50_v4.pth'

# Hyperparameters
BATCH_SIZE = 8  
LEARNING_RATE = 0.0001
NUM_EPOCHS = 15
ACCUM_STEPS = 2
DROPOUT_RATE = 0.3
IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.2
NUM_WORKERS = 4
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.1
PREDICTION_THRESHOLD = 0.3
WEIGHT_DECAY = 1e-3

class ChestXrayDataset(Dataset):
    def __init__(self, data_frame, image_dir, transform=None):
        self.data = data_frame
        self.image_dir = image_dir
        self.transform = transform
        self.labels = self.data[CLASSES].values.astype(np.float32)
        self.classes = CLASSES
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx, tries=0):
        try:
            img_name = os.path.join(self.image_dir, self.data.iloc[idx]['Image Index'])
            if not os.path.exists(img_name):
                raise FileNotFoundError(f"Image {img_name} not found")
            image = Image.open(img_name).convert('L').convert('RGB')  # Convert grayscale to RGB
            label = torch.tensor(self.labels[idx])
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            if tries < 3:
                return self.__getitem__((idx + 1) % len(self), tries + 1)
            else:
                raise RuntimeError(f"Failed to load image after 3 tries: {img_name}")

# Define data augmentation and preprocessing transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Calculate accuracy with a customizable threshold
def calculate_accuracy(preds, labels, threshold=PREDICTION_THRESHOLD):
    preds = (preds >= threshold).astype(np.float32)
    correct = (preds == labels).astype(np.float32)
    accuracy = correct.mean() * 100
    return accuracy

# Compute class weights to handle imbalanced data
def compute_class_weights(labels):
    num_samples = len(labels)
    num_classes = labels.shape[1]
    class_counts = labels.sum(axis=0)
    print("Class Distribution:", class_counts)
    weights = (num_samples - class_counts) / class_counts
    weights = np.clip(weights, 0.5, 5.0)  # Clip weights to avoid extreme values
    return torch.tensor(weights, dtype=torch.float32)

# Training and evaluation function with per-class metrics per epoch
def train_model(model, criterion, optimizer, scheduler, dataloaders, datasets, num_epochs=NUM_EPOCHS, accum_steps=ACCUM_STEPS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_classes = len(CLASSES)
    best_auc = 0.0
    patience = 5  # Early stopping patience
    epochs_no_improve = 0
    best_val_loss = float('inf')

    # Store metrics for each epoch
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []
            step = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if outputs.shape != labels.shape:
                        raise ValueError(f"Output shape {outputs.shape} does not match label shape {labels.shape}")
                    loss = criterion(outputs, labels) / accum_steps

                    if phase == 'train':
                        loss.backward()
                        step += 1
                        if step % accum_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * inputs.size(0) * accum_steps
                preds = torch.sigmoid(outputs).detach()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            epoch_loss = running_loss / len(datasets[phase])
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            # Compute overall metrics
            binary_preds = (all_preds >= PREDICTION_THRESHOLD).astype(int)
            accuracy = calculate_accuracy(all_preds, all_labels, threshold=PREDICTION_THRESHOLD)
            auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(num_classes)]
            mean_auc = np.mean(auc_scores)

            macro_f1 = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(all_labels, binary_preds, average='micro', zero_division=0)
            macro_precision = precision_score(all_labels, binary_preds, average='macro', zero_division=0)
            micro_precision = precision_score(all_labels, binary_preds, average='micro', zero_division=0)
            macro_recall = recall_score(all_labels, binary_preds, average='macro', zero_division=0)
            micro_recall = recall_score(all_labels, binary_preds, average='micro', zero_division=0)

            # Print overall metrics
            print(f'{phase.upper()} Loss: {epoch_loss:.4f} | Mean AUC: {mean_auc:.4f} | Accuracy: {accuracy:.2f}%')
            print(f'{phase.upper()} Macro - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}')
            print(f'{phase.upper()} Micro - Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}')

            # Detailed per-class metrics
            print(f"\nDetailed Metrics per Class ({phase.upper()}) - Epoch {epoch+1}:")
            report = classification_report(all_labels, binary_preds, target_names=CLASSES, zero_division=0)
            print(report)

            # AUC per class
            print(f"AUC per Class ({phase.upper()}) - Epoch {epoch+1}:")
            for i, cls in enumerate(CLASSES):
                print(f"{cls}: {auc_scores[i]:.4f}")

            # Print GPU memory usage
            if torch.cuda.is_available():
                print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            # Save best model based on validation AUC
            if phase == 'val':
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"Best model saved with AUC: {best_auc:.4f}")
                # Early stopping check
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    return model

            if phase == 'train':
                scheduler.step(epoch_loss)

            # Store metrics in history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_auc'].append(mean_auc)
                history['train_accuracy'].append(accuracy)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_auc'].append(mean_auc)
                history['val_accuracy'].append(accuracy)

    # Save training results and plot metrics
    results_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs+1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_auc': history['train_auc'],
        'val_auc': history['val_auc'],
        'train_accuracy': history['train_accuracy'],
        'val_accuracy': history['val_accuracy'],
    })
    results_csv_path = 'training_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nTraining results saved to {results_csv_path}")

    # Plot loss
    plt.figure(figsize=(10,5))
    plt.plot(results_df['epoch'], results_df['train_loss'], label='Train Loss')
    plt.plot(results_df['epoch'], results_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

    # Plot AUC
    plt.figure(figsize=(10,5))
    plt.plot(results_df['epoch'], results_df['train_auc'], label='Train AUC')
    plt.plot(results_df['epoch'], results_df['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Mean AUC')
    plt.title('Training and Validation Mean AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig('auc_plot.png')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10,5))
    plt.plot(results_df['epoch'], results_df['train_accuracy'], label='Train Accuracy')
    plt.plot(results_df['epoch'], results_df['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

    return model

# Verify images to filter out corrupted ones
def verify_images(image_dir, image_list):
    valid_images = []
    for img_name in image_list:
        try:
            img_path = os.path.join(image_dir, img_name)
            Image.open(img_path).verify()  # Raises an exception if the image is corrupted
            valid_images.append(img_name)
        except Exception as e:
            print(f"Invalid image {img_name}: {e}")
    return valid_images

# Encode labels, handling uncertain values by ignoring them
def encode_labels(labels, uncertain_handling='ignore'):
    encoded = np.zeros((len(labels), len(CLASSES)), dtype=np.float32)
    valid_indices = []
    for i, label in enumerate(labels):
        if pd.isna(label) or label == 'No Finding':
            encoded[i, CLASSES.index('No Finding')] = 1.0
            valid_indices.append(i)
        else:
            diseases = label.split('|')
            valid = True
            for disease in diseases:
                if disease in CLASSES:
                    encoded[i, CLASSES.index(disease)] = 1.0
                elif disease == '-1' or pd.isna(disease):
                    if uncertain_handling == 'ignore':
                        valid = False
            if valid:
                valid_indices.append(i)
    if uncertain_handling == 'ignore':
        return encoded[valid_indices], valid_indices
    return encoded, list(range(len(labels)))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == '__main__':
    # Load data
    data = pd.read_csv(CSV_FILE)

    # Verify images to avoid errors during training
    valid_images = verify_images(IMAGE_DIR, data['Image Index'].values)
    data = data[data['Image Index'].isin(valid_images)].reset_index(drop=True)

    # Encode labels
    encoded_labels, valid_indices = encode_labels(data['Finding Labels'], uncertain_handling='ignore')
    data = data.iloc[valid_indices].reset_index(drop=True)
    for i, cls in enumerate(CLASSES):
        data[cls] = encoded_labels[:, i]

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(encoded_labels)
    print("Class Weights:", class_weights)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=TEST_SIZE, random_state=42)

    # Create datasets
    datasets = {
        'train': ChestXrayDataset(train_data, IMAGE_DIR, transform=data_transforms['train']),
        'val': ChestXrayDataset(val_data, IMAGE_DIR, transform=data_transforms['val']),
    }

    # Create data loaders
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
    }

    # Initialize ResNet50 model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_classes = len(CLASSES)
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Set device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define loss function with class weights
    criterion = FocalLoss(pos_weight=class_weights.to(device))

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE , weight_decay=WEIGHT_DECAY)  # Add weight_decay=WEIGHT_DECAY if enabled
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    # Train the model with gradient accumulation
    trained_model = train_model(model, criterion, optimizer, scheduler, dataloaders, datasets)