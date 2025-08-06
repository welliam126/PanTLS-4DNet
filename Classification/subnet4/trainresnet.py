import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split  # Changed from KFold to train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import csv
from datetime import datetime

class BasicBlock(nn.Module):
    """Basic block for ResNet architecture"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)

class ResNetClassifier(nn.Module):
    """ResNet-based classifier for sequence features"""
    def __init__(self, input_dim=512, num_classes=4):
        super().__init__()
        self.in_channels = 64
        
        # Initial layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: (batch, seq_len, features) -> convert to (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class MedicalVolumeDataset(Dataset):
    """3D medical imaging dataset"""
    def __init__(self, root_dir, num_classes=4):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        
        # Collect data files
        for label in range(num_classes):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.endswith(".npz"):
                    file_path = os.path.join(class_dir, filename)
                    self.data.append(file_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_volume(self, vol_data):
        """Preprocess 3D medical volume data"""
        # Adjust axis order (C, H, D) -> (H, D, C)
        vol_data = np.transpose(vol_data, (1, 2, 0))
        
        # HU value normalization
        hu_min, hu_max =
        vol_data = np.clip(vol_data, hu_min, hu_max)
        vol_data = ((vol_data + 400) / 600).astype(np.float32)
        
        # Convert to tensor
        tensor = torch.tensor(vol_data)
        target_shape = (480, 480, 240)
        
        # Center crop
        h, w, d = tensor.shape
        h_start = max((h - target_shape[0]) // 2, 0)
        h_end = min(h_start + target_shape[0], h)
        w_start = max((w - target_shape[1]) // 2, 0)
        w_end = min(w_start + target_shape[1], w)
        d_start = max((d - target_shape[2]) // 2, 0)
        d_end = min(d_start + target_shape[2], d)
        
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Pad to target size
        pad_h = target_shape[0] - tensor.size(0)
        pad_w = target_shape[1] - tensor.size(1)
        pad_d = target_shape[2] - tensor.size(2)
        
        padding = (
            pad_d // 2, pad_d - pad_d // 2,
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2
        )
        
        tensor = F.pad(tensor, padding, value=-1)
        
        # Adjust dimension order (D, H, W) and add channel dimension
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        
        # Load and preprocess volume data
        volume = np.load(file_path)["volume_data"]
        tensor = self.preprocess_volume(volume)
            
        return tensor, label

def initialize_feature_extractor(device):
    """Initialize feature extractor model (placeholder implementation)"""
    class DummyFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = nn.Parameter(torch.zeros(1))
            
        def forward(self, x, return_encoded_tokens=False):
            # Generate random features (batch, depth, height, width, features)
            bs = x.size(0)
            return torch.randn(bs, 16, 16, 16, 512)
    
    model = DummyFeatureExtractor().to(device)
    model.eval()
    return model

def train_epoch(feature_extractor, classifier, train_loader, criterion, 
                optimizer, scaler, device):
    """Train for one epoch"""
    feature_extractor.eval()
    classifier.train()
    
    total_loss = 0
    all_labels = []
    all_preds = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast():
            # Extract features
            with torch.no_grad():
                features = feature_extractor(images, return_encoded_tokens=True)
                # Adjust feature shape (batch, seq_len, features)
                bs = features.size(0)
                features = features.view(bs, -1, features.size(-1))
            
            # Classification prediction
            logits = classifier(features)
            loss = criterion(logits, labels)
        
        # Gradient scaling and backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Record metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3]).flatten()
    
    return avg_loss, accuracy, f1, cm, all_labels, all_preds

def validate(feature_extractor, classifier, val_loader, criterion, device):
    """Validate model"""
    feature_extractor.eval()
    classifier.eval()
    
    val_loss = 0
    val_labels = []
    val_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                # Extract features
                features = feature_extractor(images, return_encoded_tokens=True)
                bs = features.size(0)
                features = features.view(bs, -1, features.size(-1))
                
                # Classification prediction
                logits = classifier(features)
                loss = criterion(logits, labels)
            
            # Record results
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2, 3]).flatten()
    
    return avg_val_loss, val_accuracy, val_f1, val_cm, val_labels, val_preds

def run_training(dataset, model_name, num_epochs, num_classes):
    """Run training with 6:1 train-validation split"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize feature extractor
    feature_extractor = initialize_feature_extractor(device)
    
    # Create results CSV file
    csv_file = "training_results.csv"
    cm_columns = [f"cm_{i}{j}" for i in range(num_classes) for j in range(num_classes)]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['timestamp', 'epoch', 'train_loss', 'train_acc', 'train_f1'] + cm_columns
        header += ['val_loss', 'val_acc', 'val_f1'] + cm_columns
        writer.writerow(header)
    
    # Split dataset into 6:1 train-validation ratio
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), 
        test_size=1/7,  # 1/7 of data for validation (6:1 ratio)
        shuffle=True,
        stratify=dataset.labels,
        random_state=42
    )
    
    # Create data subsets
    train_subsampler = Subset(dataset, train_idx)
    val_subsampler = Subset(dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False)
    
    # Initialize classifier
    classifier = ResNetClassifier(input_dim=512, num_classes=num_classes)
    classifier = classifier.to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = f"{model_name}_best.pth"
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc, train_f1, train_cm, _, _ = train_epoch(
            feature_extractor, classifier, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validation
        val_loss, val_acc, val_f1, val_cm, _, _ = validate(
            feature_extractor, classifier, val_loader, criterion, device
        )
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp, epoch, train_loss, train_acc, train_f1] + list(train_cm)
            row += [val_loss, val_acc, val_f1] + list(val_cm)
            writer.writerow(row)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), best_model_path)
            print(f"Saved new best model with val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    # Configuration parameters
    DATA_ROOT = "path/to/medical_volumes"  # Replace with actual path
    MODEL_NAME = "ResNetClassifier"
    NUM_EPOCHS = 
    NUM_CLASSES = 4
    
    # Create dataset
    dataset = MedicalVolumeDataset(DATA_ROOT, num_classes=NUM_CLASSES)
    
    # Run training with 6:1 train-validation split
    run_training(
        dataset=dataset,
        model_name=MODEL_NAME,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES
    )