import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler
from CT_CLIP.ct_clip import CTCLIP
from transformer_maskgit import CTViT

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MedicalVolumeDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=4):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Iterate through all class directories
        for label in range(num_classes):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
                
            # Collect all NPZ files
            for filename in os.listdir(class_dir):
                if filename.endswith(".npz"):
                    file_path = os.path.join(class_dir, filename)
                    self.data.append(file_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_volume(self, vol_data):
        """Preprocess 3D medical imaging volume data"""
        # Adjust axis order
        vol_data = np.transpose(vol_data, (1, 2, 0))
        
        # HU value normalization
        hu_min, hu_max = 
        vol_data = np.clip(vol_data, hu_min, hu_max)
        vol_data = ((vol_data + 400) / 600).astype(np.float32)
        
        # Convert to tensor
        tensor = torch.tensor(vol_data)
        target_shape = (480, 480, 240)
        
        # Center crop or pad
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
        # Calculate crop range
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        # Perform cropping
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Calculate padding amounts
        pad_h = (dh - tensor.size(0))
        pad_w = (dw - tensor.size(1))
        pad_d = (dd - tensor.size(2))
        
        # Symmetric padding
        padding = (
            pad_d // 2, pad_d - pad_d // 2,
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2
        )
        
        # Apply padding
        tensor = F.pad(tensor, padding, value=-1)
        
        # Adjust dimension order (D, H, W)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)  # Add channel dimension
        
        return tensor
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        
        # Load NPZ file
        volume = np.load(file_path)["volume_data"]
        
        # Preprocess volume data
        tensor = self.preprocess_volume(volume)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label

def initialize_model():
    """Initialize vision transformer model"""
    # Example: Using publicly available text model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    
    
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
    
 
    model = CTCLIP(
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
    
 
    return model.visual_transformer
    

def train_model(data_dir, num_classes=4, val_split=0.1667):
    """Train classification model with 6:1 train-validation split"""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create full dataset
    full_dataset = MedicalVolumeDataset(data_dir, num_classes=num_classes)
    
    # Split dataset into train and validation (6:1 ratio)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize feature extraction model
    feature_extractor = initialize_model().to(device)
    feature_extractor.eval()
    
    # Initialize classifier
    input_dim = 512  # Feature dimension
    classifier = MLPClassifier(input_dim, num_classes).to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Extract features
                with torch.no_grad():
                    features = feature_extractor(images, return_encoded_tokens=True)
                    # Global average pooling
                    features = torch.mean(features, dim=[1, 2, 3])
                
                # Classification prediction
                logits = classifier(features)
                loss = criterion(logits, labels)
            
            # Gradient scaling and backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Record metrics
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Validation phase
        classifier.eval()
        val_loss = 0
        val_labels = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with autocast():
                    features = feature_extractor(images, return_encoded_tokens=True)
                    features = torch.mean(features, dim=[1, 2, 3])
                    logits = classifier(features)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # Save model
    torch.save(classifier.state_dict(), "trained_classifier.pth")
    print("Training complete. Model saved as trained_classifier.pth")

if __name__ == "__main__":
    # Example path - replace with actual path in use
    data_dir = "path/to/your/data_directory"
    
    # Start training with 6:1 train-validation split
    train_model(data_dir, num_classes=4)