import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import sys
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import pandas as pd

# Add current directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom module imports (need to provide implementations of these modules)
from ct_clip import CTCLIP
from transformer_maskgit import CTViT

# Configuration parameters
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 50,
    "batch_size": 4,
    "num_workers": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_classes": 4,
    "input_dim": 512,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "data_root": "/path/to/your/dataset",  # User needs to set this
    "pretrained_path": "/path/to/pretrained/model",  # User needs to set this
    "output_dir": "/path/to/output",  # User needs to set this
    "test_size": 0.1428,  # 1/7 of data for validation (6:1 ratio)
    "random_state": 42  # For reproducibility
}

# Ensure output directory exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)

class ViTClassifier(nn.Module):
    """Transformer-based classifier"""
    def __init__(self, input_dim=512, num_classes=4, hidden_dim=256, num_heads=8, num_layers=2):
        super(ViTClassifier, self).__init__()
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1)]
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Use first token for classification
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)

class CTDataset(Dataset):
    """CT scan dataset"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        
        # Load data paths
        for label in range(CONFIG["num_classes"]):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.endswith(".npz"):
                    self.data.append(os.path.join(class_dir, filename))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load and process CT scan
        scan = np.load(self.data[idx])
        scan_data = scan["tensor"][0]
        return self.preprocess_scan(scan_data), self.labels[idx]
    
    def preprocess_scan(self, scan):
        """Preprocess CT scan data"""
        # Adjust orientation and normalize
        scan = np.transpose(scan, (1, 2, 0))
        scan = np.clip(scan, ,)
        scan = (((scan + 400) / 600)).astype(np.float32)
        
        # Convert to tensor
        tensor = torch.tensor(scan)
        target_shape = (480, 480, 240)
        
        # Crop or pad to target size
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
        # Center crop
        h_start, w_start, d_start = max(0, (h - dh) // 2), max(0, (w - dw) // 2), max(0, (d - dd) // 2)
        tensor = tensor[
            h_start:h_start+dh, 
            w_start:w_start+dw, 
            d_start:d_start+dd
        ]
        
        # Center padding
        pad_h = max(0, dh - tensor.size(0))
        pad_w = max(0, dw - tensor.size(1))
        pad_d = max(0, dd - tensor.size(2))
        
        tensor = F.pad(tensor, (
            pad_d//2, pad_d - pad_d//2,
            pad_w//2, pad_w - pad_w//2,
            pad_h//2, pad_h - pad_h//2
        ), value=-1)
        
        # Adjust dimension order [Depth, Height, Width] -> [1, Depth, Height, Width]
        return tensor.permute(2, 0, 1).unsqueeze(0)

def load_pretrained_model(device):
    """Load pretrained model"""
    # Initialize tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    
    # Initialize image encoder
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )
    
    # Initialize CTCLIP model
    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )
    
    # Load pretrained weights
    clip.load(CONFIG["pretrained_path"])
    
    # Use only the visual transformer part
    model = clip.visual_transformer
    del clip
    
    # Set to evaluation mode
    model.eval().to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model

def train_epoch(model, classifier, loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    classifier.train()
    total_loss = 0
    metrics = {"labels": [], "preds": [], "probs": []}
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast():
            # Extract features
            with torch.no_grad():
                features = model(images, return_encoded_tokens=True)
                if isinstance(features, tuple):
                    features = features[0]
                features = features.view(images.size(0), -1, CONFIG["input_dim"])
            
            # Classification
            logits = classifier(features)
            loss = criterion(logits, labels)
        
        # Mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Record metrics
        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        metrics["labels"].extend(labels.cpu().numpy())
        metrics["preds"].extend(logits.argmax(dim=1).cpu().numpy())
        metrics["probs"].extend(probs.cpu().detach().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(metrics["labels"], metrics["preds"])
    f1 = f1_score(metrics["labels"], metrics["preds"], average='weighted')
    
    # Multi-class AUC calculation
    try:
        auc = roc_auc_score(metrics["labels"], metrics["probs"], multi_class='ovr')
    except:
        auc = 0.5
    
    return avg_loss, accuracy, auc, f1

def validate(model, classifier, loader, criterion, device):
    """Validate model"""
    classifier.eval()
    total_loss = 0
    metrics = {"labels": [], "preds": [], "probs": []}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                # Extract features
                features = model(images, return_encoded_tokens=True)
                if isinstance(features, tuple):
                    features = features[0]
                features = features.view(images.size(0), -1, CONFIG["input_dim"])
                
                # Classification
                logits = classifier(features)
                loss = criterion(logits, labels)
            
            # Record metrics
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            metrics["labels"].extend(labels.cpu().numpy())
            metrics["preds"].extend(logits.argmax(dim=1).cpu().numpy())
            metrics["probs"].extend(probs.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(metrics["labels"], metrics["preds"])
    f1 = f1_score(metrics["labels"], metrics["preds"], average='weighted')
    
    # Multi-class AUC calculation
    try:
        auc = roc_auc_score(metrics["labels"], metrics["probs"], multi_class='ovr')
    except:
        auc = 0.5
    
    return avg_loss, accuracy, auc, f1

def run_training(dataset):
    """Run training with 6:1 train-validation split"""
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    # Load pretrained model
    model = load_pretrained_model(device)
    
    # Split dataset into train and validation (6:1 ratio)
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=dataset.labels
    )
    
    # Create data subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    
    # Create data loaders
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
    
    # Initialize classifier
    classifier = ViTClassifier(
        input_dim=CONFIG["input_dim"],
        num_classes=CONFIG["num_classes"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)
    
    # Optimizer and loss function
    optimizer = torch.optim.AdamW(
        classifier.parameters(), 
        lr=CONFIG["learning_rate"], 
        weight_decay=CONFIG["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Results DataFrame
    results_df = pd.DataFrame(columns=[
        'epoch', 
        'train_loss', 'train_acc', 'train_auc', 'train_f1',
        'val_loss', 'val_acc', 'val_auc', 'val_f1'
    ])
    
    # Results CSV path
    csv_path = os.path.join(CONFIG["output_dir"], "training_results.csv")
    
    # Best model path
    best_model_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
    best_val_auc = 0.0
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss, train_acc, train_auc, train_f1 = train_epoch(
            model, classifier, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_f1 = validate(
            model, classifier, val_loader, criterion, device
        )
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        
        # Save results
        epoch_data = {
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
            torch.save(
                classifier.module.state_dict() if hasattr(classifier, 'module') else classifier.state_dict(),
                best_model_path
            )
            print(f"Saved new best model with AUC: {best_val_auc:.4f}")
    
    print("Training completed!")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    # Load dataset
    dataset = CTDataset(CONFIG["data_root"])
    print(f"Dataset size: {len(dataset)}")
    
    # Run training with 6:1 train-validation split
    run_training(dataset)