import os
import json  
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast

# 导入自定义模块（需要确保在代码库中可用）
# from CT_CLIP.ct_clip import CTCLIP
# from transformer_maskgit import CTViT

class BasicBlock(nn.Module):
    """ResNet基础块"""
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
    """用于序列特征的ResNet分类器"""
    def __init__(self, input_dim=512, num_classes=4):
        super().__init__()
        self.in_channels = 64
        
        # 初始层
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet块
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        # 输入形状: (batch, seq_len, features) -> 转换为 (batch, features, seq_len)
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
    """3D医学影像测试数据集"""
    def __init__(self, root_dir, num_classes=4):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        
        # 收集数据文件
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
        """预处理3D医学影像体积数据"""
        # 调整轴顺序 (C, H, D) -> (H, D, C)
        vol_data = np.transpose(vol_data, (1, 2, 0))
        
        # HU值标准化
        hu_min, hu_max = -1000, 200
        vol_data = np.clip(vol_data, hu_min, hu_max)
        vol_data = ((vol_data - hu_min) / (hu_max - hu_min)).astype(np.float32)
        
        # 转换为张量
        tensor = torch.tensor(vol_data)
        target_shape = (480, 480, 240)
        
        # 中心裁剪
        h, w, d = tensor.shape
        h_start = max((h - target_shape[0]) // 2, 0)
        h_end = min(h_start + target_shape[0], h)
        w_start = max((w - target_shape[1]) // 2, 0)
        w_end = min(w_start + target_shape[1], w)
        d_start = max((d - target_shape[2]) // 2, 0)
        d_end = min(d_start + target_shape[2], d)
        
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # 填充到目标尺寸
        pad_h = target_shape[0] - tensor.size(0)
        pad_w = target_shape[1] - tensor.size(1)
        pad_d = target_shape[2] - tensor.size(2)
        
        padding = (
            pad_d // 2, pad_d - pad_d // 2,
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2
        )
        
        tensor = F.pad(tensor, padding, value=0)
        
        # 调整维度顺序 (D, H, W) 并添加通道维度
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        
        # 加载并预处理体积数据
        volume = np.load(file_path)["volume_data"]
        tensor = self.preprocess_volume(volume)
            
        return tensor, label, file_path

def initialize_feature_extractor(device):
    """初始化特征提取模型（占位实现）"""
    class DummyFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = nn.Parameter(torch.zeros(1))
            
        def forward(self, x, return_encoded_tokens=False):
            # 生成随机特征 (batch, depth, height, width, features)
            bs = x.size(0)
            return torch.randn(bs, 16, 16, 16, 512)
    
    model = DummyFeatureExtractor().to(device)
    model.eval()
    return model

def run_inference(classifier_path, test_loader, device, output_dir):
    """运行推理过程"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化特征提取器
    feature_extractor = initialize_feature_extractor(device)
    
    # 初始化分类器
    classifier = ResNetClassifier(input_dim=512, num_classes=4).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    
    results = []
    
    with torch.no_grad():
        for volumes, labels, file_paths in tqdm(test_loader, desc="Processing"):
            volumes = volumes.to(device)
            
            with autocast():
                # 提取特征
                features = feature_extractor(volumes, return_encoded_tokens=True)
                
                # 调整特征形状为序列 (batch, seq_len, features)
                batch_size = features.size(0)
                features = features.view(batch_size, -1, features.size(-1))
                
                # 分类预测
                logits = classifier(features)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
            
            # 收集结果
            for path, true_label, pred, prob in zip(file_paths, labels, preds.cpu(), probs.cpu()):
                results.append({
                    'file_path': path,
                    'true_label': int(true_label),
                    'predicted_label': int(pred),
                    'probabilities': prob.tolist()
                })
    
    # 保存结果
    output_file = os.path.join(output_dir, "inference_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference results saved to {output_file}")
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 配置路径（实际使用需替换）
    TEST_DATA_DIR = "path/to/test_data"  # 测试数据集路径
    CLASSIFIER_PATH = "path/to/trained_classifier.pth"  # 训练好的分类器路径
    OUTPUT_DIR = "path/to/results"  # 输出目录
    
    # 创建测试数据集
    test_dataset = MedicalVolumeDataset(TEST_DATA_DIR)
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 运行推理
    results = run_inference(CLASSIFIER_PATH, test_loader, device, OUTPUT_DIR)