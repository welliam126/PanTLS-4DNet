import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import sys
from torch.cuda.amp import autocast

# Add current directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration parameters
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 4,
    "input_dim": 512,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "test_data_root": "/path/to/test/dataset",  # User needs to set this
    "pretrained_path": "/path/to/pretrained/ct_model",  # User needs to set this
    "mlp_model_path": "/path/to/mlp/model",  # User needs to set this
    "output_dir": "/path/to/output"  # User needs to set this
}

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

class CTTestDataset(Dataset):
    """CT scan test dataset"""
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
    
    def __getitem__(self, idx):
        # Load and process CT scan
        scan = np.load(self.data[idx])
        scan_data = scan["tensor"][0]
        return self.preprocess_scan(scan_data), self.labels[idx], self.data[idx]
    
    def preprocess_scan(self, scan):
        """Preprocess CT scan data - consistent with training"""
        # Adjust orientation and normalize
        scan = np.transpose(scan, (1, 2, 0))
        
        # HU value clipping and normalization
        hu_min, hu_max = -1000, 200
        scan = np.clip(scan, hu_min, hu_max)
        scan = ((scan - hu_min) / (hu_max - hu_min)).astype(np.float32)
        
        # Convert to tensor
        tensor = torch.tensor(scan)
        target_shape = (480, 480, 240)
        
        # Center crop
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
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
        ), value=0)  # Use 0 for padding
        
        # Adjust dimension order [Depth, Height, Width] -> [1, Depth, Height, Width]
        return tensor.permute(2, 0, 1).unsqueeze(0)

def load_pretrained_model(device):
    """Load pretrained CT feature extraction model"""
    
    
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
    return model

def run_inference(test_loader, device, output_dir):
    """Run inference and save results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pretrained CT feature extractor
    feature_extractor = load_pretrained_model(device)
    
    # Initialize classifier
    classifier = ViTClassifier(
        input_dim=CONFIG["input_dim"],
        num_classes=CONFIG["num_classes"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)
    
    # Load trained classifier weights
    classifier.load_state_dict(torch.load(CONFIG["mlp_model_path"], map_location=device))
    classifier.eval()
    
    results = []
    
    with torch.no_grad():
        for images, labels, image_paths in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            
            with autocast():
                # Extract features
                features = feature_extractor(images, return_encoded_tokens=True)
                
                # Reshape features (batch_size, seq_len, feature_dim)
                batch_size = features.size(0)
                features = features.view(batch_size, -1, CONFIG["input_dim"])
                
                # Classify
                logits = classifier(features)
            
            # Calculate probabilities
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Collect results
            for i in range(len(image_paths)):
                results.append({
                    'image_path': image_paths[i],
                    'true_label': int(labels[i]),
                    'predicted_label': int(preds[i]),
                    'probabilities': probs[i].cpu().numpy().tolist()
                })
    
    # Save results
    output_path = os.path.join(output_dir, "inference_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference completed! Results saved to {output_path}")
    return results

def analyze_results(results):
    """Analyze inference results and print statistics"""
    true_labels = [item['true_label'] for item in results]
    pred_labels = [item['predicted_label'] for item in results]
    
    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
    
    print(f"\nResults Analysis:")
    print(f"Total Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Class statistics
    class_stats = {i: {'correct': 0, 'total': 0} for i in range(CONFIG["num_classes"])}
    
    for item in results:
        class_stats[item['true_label']]['total'] += 1
        if item['true_label'] == item['predicted_label']:
            class_stats[item['true_label']]['correct'] += 1
    
    print("\nClass-wise Accuracy:")
    for class_id, stats in class_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"Class {class_id}: {acc:.4f} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    # Set device
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = CTTestDataset(CONFIG["test_data_root"])
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=12
    )
    
    # Run inference
    results = run_inference(test_loader, device, CONFIG["output_dir"])
    
    # Analyze results
    analyze_results(results)