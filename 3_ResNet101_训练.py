"""
🌱 植物幼苗分类 - ResNet-101训练
===================================
功能：
1. 构建ResNet-101模型（预训练+迁移学习）
2. 处理数据不平衡（加权损失/Focal Loss）
3. 数据增强
4. 训练循环
5. 学习率调度
6. 保存最佳模型

作者：霸道志伟
日期：2026-02-15
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm


# ==================== 配置 ====================
class Config:
    """训练配置"""
    # 数据路径
    data_dir = Path('./data/split')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    weights_file = Path('./data/class_weights.json')
    
    # 输出路径
    output_dir = Path('./output')
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    # 模型参数
    model_name = 'resnet101'
    num_classes = 12
    pretrained = True
    
    # 训练参数
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # 数据增强
    image_size = 224
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 其他
    num_workers = 4  # Windows下建议设为0，避免多进程问题
    save_every = 5  # 每N个epoch保存一次
    early_stop_patience = 10  # 早停耐心值
    
    # 损失函数类型: 'ce' (CrossEntropy), 'weighted_ce', 'focal'
    loss_type = 'weighted_ce'
    
    def __init__(self):
        """创建输出目录"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ==================== 数据集 ====================
class PlantSeedlingsDataset(Dataset):
    """植物幼苗数据集"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据转换
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 获取所有类别
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 收集所有图像路径和标签
        self.samples = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            cls_idx = self.class_to_idx[cls]
            
            for img_path in cls_dir.glob('*.png'):
                self.samples.append((img_path, cls_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== 数据增强 ====================
def get_transforms(config, mode='train'):
    """
    获取数据转换
    
    Args:
        config: 配置对象
        mode: 'train' 或 'val'
    
    Returns:
        transforms组合
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225]     # ImageNet标准差
            )
        ])
    else:  # val
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ==================== 模型 ====================
def build_model(config):
    """
    构建ResNet-101模型
    
    Args:
        config: 配置对象
    
    Returns:
        model: PyTorch模型
    """
    print(f"\n{'='*60}")
    print(f"🏗️ 构建 {config.model_name.upper()} 模型")
    print(f"{'='*60}")
    
    # 加载预训练模型
    if config.pretrained:
        print(f"✅ 加载预训练权重（ImageNet）")
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        model = models.resnet101(weights=weights)
    else:
        print(f"⚠️ 从头开始训练")
        model = models.resnet101(weights=None)
    
    # 修改最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.num_classes)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n【模型信息】")
    print(f"  架构: ResNet-101")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  输入尺寸: {config.image_size}x{config.image_size}")
    print(f"  输出类别: {config.num_classes}")
    
    return model


# ==================== 损失函数 ====================
class FocalLoss(nn.Module):
    """Focal Loss - 用于处理类别不平衡"""
    
    def __init__(self, alpha=None, gamma=2.0):
        """
        Args:
            alpha: 类别权重
            gamma: 聚焦参数
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def get_loss_function(config):
    """
    获取损失函数
    
    Args:
        config: 配置对象
    
    Returns:
        loss_fn: 损失函数
    """
    print(f"\n{'='*60}")
    print(f"📊 配置损失函数")
    print(f"{'='*60}")
    
    if config.loss_type == 'ce':
        print("✅ 使用标准交叉熵损失 (CrossEntropyLoss)")
        return nn.CrossEntropyLoss()
    
    elif config.loss_type == 'weighted_ce':
        print("✅ 使用加权交叉熵损失 (Weighted CrossEntropyLoss)")
        
        # 加载类别权重
        if config.weights_file.exists():
            with open(config.weights_file, 'r', encoding='utf-8') as f:
                class_weights_dict = json.load(f)
            
            # 转换为Tensor
            dataset = PlantSeedlingsDataset(config.train_dir)
            class_names = dataset.classes
            weights = [class_weights_dict[cls] for cls in class_names]
            class_weights = torch.FloatTensor(weights).to(config.device)
            
            print(f"  加载类别权重: {config.weights_file}")
            print(f"  权重范围: {class_weights.min():.4f} - {class_weights.max():.4f}")
            
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            print(f"  ⚠️ 未找到权重文件，使用标准交叉熵")
            return nn.CrossEntropyLoss()
    
    elif config.loss_type == 'focal':
        print("✅ 使用Focal Loss")
        
        # 可选：加载类别权重
        alpha = None
        if config.weights_file.exists():
            with open(config.weights_file, 'r', encoding='utf-8') as f:
                class_weights_dict = json.load(f)
            
            dataset = PlantSeedlingsDataset(config.train_dir)
            class_names = dataset.classes
            weights = [class_weights_dict[cls] for cls in class_names]
            alpha = torch.FloatTensor(weights).to(config.device)
            print(f"  使用类别权重")
        
        return FocalLoss(alpha=alpha, gamma=2.0)
    
    else:
        raise ValueError(f"未知的损失类型: {config.loss_type}")


# ==================== 训练和验证 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """验证"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ==================== 主训练函数 ====================
def train(config):
    """主训练函数"""
    print("🌱" * 30)
    print("开始训练 ResNet-101 模型")
    print("🌱" * 30)
    
    # 打印配置
    print(f"\n【训练配置】")
    print(f"  设备: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  权重衰减: {config.weight_decay}")
    print(f"  损失函数: {config.loss_type}")
    print(f"  图像尺寸: {config.image_size}x{config.image_size}")
    
    # 创建数据集
    print(f"\n{'='*60}")
    print(f"📁 加载数据集")
    print(f"{'='*60}")
    
    train_dataset = PlantSeedlingsDataset(
        config.train_dir,
        transform=get_transforms(config, mode='train')
    )
    
    val_dataset = PlantSeedlingsDataset(
        config.val_dir,
        transform=get_transforms(config, mode='val')
    )
    
    print(f"  训练集: {len(train_dataset)} 张图像")
    print(f"  验证集: {len(val_dataset)} 张图像")
    print(f"  类别数: {len(train_dataset.classes)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下设为0
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    # 构建模型
    model = build_model(config)
    model = model.to(config.device)
    
    # 损失函数
    criterion = get_loss_function(config)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(config.log_dir)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 训练循环
    print(f"\n{'='*60}")
    print(f"🚀 开始训练")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.device, epoch
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 打印
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': vars(config)
            }
            
            best_model_path = config.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"  ✅ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # 定期保存
        if epoch % config.save_every == 0:
            checkpoint_path = config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  💾 保存检查点: {checkpoint_path.name}")
        
        # 早停
        if patience_counter >= config.early_stop_patience:
            print(f"\n⏸️ 早停触发！验证准确率已连续{config.early_stop_patience}个epoch没有提升")
            break
        
        print("-" * 60)
    
    # 训练结束
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 训练完成！")
    print(f"{'='*60}")
    print(f"  总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"  最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  最佳模型保存在: {config.checkpoint_dir / 'best_model.pth'}")
    
    # 保存训练历史
    history_path = config.output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  训练历史保存在: {history_path}")
    
    writer.close()
    
    return history


def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 检查数据目录
    if not config.train_dir.exists():
        print(f"❌ 错误: 找不到训练数据目录 {config.train_dir}")
        print("请先运行 1_数据准备.py 来准备数据")
        return
    
    # 开始训练
    history = train(config)
    
    print(f"\n下一步: 运行 4_模型评估与混淆矩阵.py 来评估模型")


if __name__ == '__main__':
    main()
