import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

class VGGClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=4, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        # Global average pooling
        x = x.mean(dim=1)
        features = self.features(x)
        return self.classifier(features)

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
        # Adjust axis order
        vol_data = np.transpose(vol_data, (1, 2, 0))
        
        # HU value normalization
        hu_min, hu_max = -1000, 1000  # Typical HU value range for CT scans
        vol_data = np.clip(vol_data, hu_min, hu_max)
        vol_data = ((vol_data - hu_min) / (hu_max - hu_min)).astype(np.float32)
        
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
        
        # Adjust dimension order and add channel dimension
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
                # Reshape features to sequence (batch, seq_len, features)
                batch_size = features.size(0)
                features = features.view(batch_size, -1, features.size(-1))
            
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
    
    # Calculate AUC-ROC (handling multi-class case)
    try:
        auc_roc = roc_auc_score(all_labels, 
                               F.one_hot(torch.tensor(all_preds), num_classes=4).numpy(),
                               multi_class='ovr')
    except Exception:
        auc_roc = 0.5
    
    return avg_loss, accuracy, auc_roc, f1

def validate(feature_extractor, classifier, val_loader, criterion, device):
    """Validate the model"""
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
                batch_size = features.size(0)
                features = features.view(batch_size, -1, features.size(-1))
                
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
    
    # Calculate AUC-ROC
    try:
        val_auc_roc = roc_auc_score(val_labels, 
                                   F.one_hot(torch.tensor(val_preds), num_classes=4).numpy(),
                                   multi_class='ovr')
    except Exception:
        val_auc_roc = 0.5
    
    return avg_val_loss, val_accuracy, val_auc_roc, val_f1

def run_training(dataset, model_name, num_epochs, num_classes, results_dir):
    """Run training and validation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize feature extractor
    feature_extractor = initialize_feature_extractor(device)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Split dataset in 6:1 ratio
    train_size = int(0.857 * len(dataset))  # 6/7 ≈ 0.857
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize classifier
    classifier = VGGClassifier(num_classes=num_classes).to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    best_val_auc = 0.0
    best_model_path = os.path.join(results_dir, f"{model_name}_best.pth")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc, train_auc, train_f1 = train_epoch(
            feature_extractor, classifier, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validation
        val_loss, val_acc, val_auc, val_f1 = validate(
            feature_extractor, classifier, val_loader, criterion, device
        )
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(classifier.state_dict(), best_model_path)
            print(f"Saved new best model with Val AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    # Configuration parameters
    DATA_DIR = "path/to/medical_volumes"  # Dataset path
    MODEL_NAME = "VGGClassifier"
    NUM_EPOCHS =  # Number of training epochs
    NUM_CLASSES = 4
    RESULTS_DIR = "path/to/results"  # Directory to save results
    
    # Create dataset
    dataset = MedicalVolumeDataset(DATA_DIR, num_classes=NUM_CLASSES)
    
    # Run training
    run_training(
        dataset=dataset,
        model_name=MODEL_NAME,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES,
        results_dir=RESULTS_DIR
    )