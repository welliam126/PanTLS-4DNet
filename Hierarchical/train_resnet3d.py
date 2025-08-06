import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import nibabel as nib    
from resnet3d import resnet10

# Configuration parameters
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 1000,
    "k_folds": 5,
    "batch_size": 12,
    "num_workers": 12,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_classes": 4,
    "data_root": "/path/to/your/dataset",  # Set by user
    "output_dir": "/path/to/output",  # Set by user
    "model_name": "resnet3d",
    "target_shape": (480, 480, 240),  # Target volume dimensions
    "hu_range": (,),  # HU value range
    "normalization": (-400, 600)  # Normalization parameters
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

class CTDataset(Dataset):
    """Dataset class for CT scans"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        
        # Load data paths
        for label in range(CONFIG["num_classes"]):
            class_dir = os.path.join(root_dir, str(label))
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(".npz"):
                        self.data.append(os.path.join(class_dir, filename))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_scan(self, scan_data):
        """Preprocess CT scan data"""
        # Adjust orientation
        scan_data = np.transpose(scan_data, (1, 2, 0))
        
        tensor = torch.tensor(scan_data).float()
        
        # HU normalization
        hu_min, hu_max = CONFIG["hu_range"]
        tensor = torch.clamp(tensor, hu_min, hu_max)
        norm_min, norm_div = CONFIG["normalization"]
        tensor = (tensor - norm_min) / norm_div
        
        dh, dw, dd = CONFIG["target_shape"]
        
        # Center cropping
        h, w, d = tensor.shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Center padding
        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = F.pad(tensor, 
                      (pad_d_before, pad_d_after, 
                       pad_w_before, pad_w_after, 
                       pad_h_before, pad_h_after), 
                      value=0)

        # Dimension adjustment: [D, H, W] -> [C, D, H, W]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        # Load CT scan
        npz_file = np.load(self.data[idx])
        scan_data = npz_file['tensor'][0]
        label = self.labels[idx]
        
        tensor = self.preprocess_scan(scan_data)
        return tensor, label

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics = {"labels": [], "preds": [], "probs": []}
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        
        metrics["labels"].extend(labels.cpu().numpy())
        metrics["preds"].extend(logits.argmax(dim=1).cpu().numpy())
        metrics["probs"].extend(probs.cpu().detach().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(metrics["labels"], metrics["preds"])
    f1 = f1_score(metrics["labels"], metrics["preds"], average='weighted')
    
    # Multi-class AUC
    try:
        auc = roc_auc_score(metrics["labels"], metrics["probs"], multi_class='ovr')
    except:
        auc = 0.5
    
    return avg_loss, accuracy, auc, f1

def validate(model, val_loader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    metrics = {"labels": [], "preds": [], "probs": []}
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            
            metrics["labels"].extend(labels.cpu().numpy())
            metrics["preds"].extend(logits.argmax(dim=1).cpu().numpy())
            metrics["probs"].extend(probs.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(metrics["labels"], metrics["preds"])
    f1 = f1_score(metrics["labels"], metrics["preds"], average='weighted')
    
    # Multi-class AUC
    try:
        auc = roc_auc_score(metrics["labels"], metrics["probs"], multi_class='ovr')
    except:
        auc = 0.5
    
    return avg_loss, accuracy, auc, f1

def run_cross_validation(dataset):
    """Execute k-fold cross-validation"""
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    kfold = KFold(n_splits=CONFIG["k_folds"], shuffle=True, random_state=42)
    
    results_df = pd.DataFrame(columns=[
        'fold', 'epoch', 
        'train_loss', 'train_acc', 'train_auc', 'train_f1',
        'val_loss', 'val_acc', 'val_auc', 'val_f1'
    ])
    
    csv_path = os.path.join(CONFIG["output_dir"], "training_results.csv")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nStarting Fold {fold + 1}/{CONFIG['k_folds']}")
        
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_set, 
            batch_size=CONFIG["batch_size"], 
            shuffle=True, 
            num_workers=CONFIG["num_workers"]
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=CONFIG["batch_size"], 
            shuffle=False, 
            num_workers=CONFIG["num_workers"]
        )
        
  
        model = resnet10(num_seg_classes=CONFIG["num_classes"])
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=CONFIG["learning_rate"], 
            weight_decay=CONFIG["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        best_model_path = os.path.join(
            CONFIG["output_dir"], 
            f"{CONFIG['model_name']}_best_model_fold{fold+1}.pth"
        )
        best_val_auc = 0.0
        
        for epoch in range(CONFIG["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
            
            train_loss, train_acc, train_auc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            
            val_loss, val_acc, val_auc, val_f1 = validate(
                model, val_loader, criterion, device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
            
            # Save epoch results
            epoch_data = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_auc': train_auc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1
            }
            results_df = pd.concat([results_df, pd.DataFrame([epoch_data])], ignore_index=True)
            results_df.to_csv(csv_path, index=False)
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, best_model_path)
                print(f"Saved new best model with AUC: {best_val_auc:.4f}")
    
    print("Cross-validation completed!")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    dataset = CTDataset(CONFIG["data_root"])
    print(f"Loaded {len(dataset)} CT scans")
    run_cross_validation(dataset)