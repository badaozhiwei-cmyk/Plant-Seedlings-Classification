"""
🌱 植物幼苗分类 - 模型评估与混淆矩阵
===================================
功能：
1. 加载训练好的模型
2. 在验证集上评估
3. 生成混淆矩阵
4. 计算各类别指标（Precision, Recall, F1-Score）
5. 错误分析和可视化
6. 分析哪些类别容易混淆

作者：霸道志伟
日期：2026-02-15
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

# 导入训练脚本中的数据集类
import sys
sys.path.append(str(Path(__file__).parent))
from importlib import import_module

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path, data_dir, device=None):
        """
        Args:
            model_path: 模型checkpoint路径
            data_dir: 验证数据目录
            device: 计算设备
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输出目录
        self.output_dir = Path('./output/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别名称（中英文对照）
        self.class_names_cn = {
            'Black-grass': '黑麦草',
            'Charlock': '野芥菜',
            'Cleavers': '猪殃殃',
            'Common Chickweed': '繁缕',
            'Common wheat': '小麦',
            'Fat Hen': '藜',
            'Loose Silky-bent': '丝草',
            'Maize': '玉米',
            'Scentless Mayweed': '无味春黄菊',
            'Shepherds Purse': '荠菜',
            'Small-flowered Cranesbill': '小花老鹳草',
            'Sugar beet': '甜菜'
        }
        
        # 加载模型
        self.model, self.checkpoint = self.load_model()
        
        # 加载数据
        self.dataloader, self.class_names = self.load_data()
        
    def load_model(self):
        """加载训练好的模型"""
        print(f"{'='*60}")
        print(f"📦 加载模型")
        print(f"{'='*60}")
        
        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        print(f"  模型路径: {self.model_path}")
        print(f"  训练轮次: Epoch {checkpoint['epoch']}")
        print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
        print(f"  验证损失: {checkpoint['val_loss']:.4f}")
        
        # 重建模型
        model = models.resnet101(weights=None)
        num_features = model.fc.in_features
        num_classes = checkpoint['config']['num_classes']
        model.fc = nn.Linear(num_features, num_classes)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ 模型加载成功！")
        
        return model, checkpoint
    
    def load_data(self):
        """加载验证数据"""
        print(f"\n{'='*60}")
        print(f"📁 加载验证数据")
        print(f"{'='*60}")
        
        # 导入数据集类
        try:
            train_module = import_module('3_ResNet101_训练')
            PlantSeedlingsDataset = train_module.PlantSeedlingsDataset
        except:
            # 如果导入失败，使用本地定义
            from torch.utils.data import Dataset
            
            class PlantSeedlingsDataset(Dataset):
                def __init__(self, data_dir, transform=None):
                    self.data_dir = Path(data_dir)
                    self.transform = transform
                    self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
                    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
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
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, label
        
        # 数据转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 创建数据集
        dataset = PlantSeedlingsDataset(self.data_dir, transform=transform)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        print(f"  数据目录: {self.data_dir}")
        print(f"  样本数量: {len(dataset)}")
        print(f"  类别数量: {len(dataset.classes)}")
        
        return dataloader, dataset.classes
    
    def predict(self):
        """在验证集上进行预测"""
        print(f"\n{'='*60}")
        print(f"🔮 模型预测")
        print(f"{'='*60}")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="预测中"):
                images = images.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n✅ 预测完成！")
        print(f"  总准确率: {accuracy * 100:.2f}%")
        
        return all_preds, all_labels, all_probs
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        print(f"\n{'='*60}")
        print(f"📊 生成混淆矩阵")
        print(f"{'='*60}")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 归一化（按行）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 创建中文类别名
        class_labels_cn = [self.class_names_cn.get(cls, cls) for cls in self.class_names]
        
        # 绘制两个图：原始计数和归一化
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 图1: 原始计数
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels_cn,
                   yticklabels=class_labels_cn,
                   ax=axes[0], cbar_kws={'label': '样本数量'})
        axes[0].set_ylabel('真实类别', fontsize=12)
        axes[0].set_xlabel('预测类别', fontsize=12)
        axes[0].set_title('混淆矩阵（原始计数）', fontsize=14, fontweight='bold')
        
        # 图2: 归一化
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_labels_cn,
                   yticklabels=class_labels_cn,
                   ax=axes[1], cbar_kws={'label': '比例'}, vmin=0, vmax=1)
        axes[1].set_ylabel('真实类别', fontsize=12)
        axes[1].set_xlabel('预测类别', fontsize=12)
        axes[1].set_title('混淆矩阵（归一化）', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存到: {output_path}")
        plt.show()
        
        return cm, cm_normalized
    
    def analyze_per_class_metrics(self, y_true, y_pred):
        """分析每个类别的指标"""
        print(f"\n{'='*60}")
        print(f"📈 每类别性能分析")
        print(f"{'='*60}\n")
        
        # 计算指标
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # 创建DataFrame
        metrics_df = pd.DataFrame({
            '类别': self.class_names,
            '中文名': [self.class_names_cn.get(cls, cls) for cls in self.class_names],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            '样本数': support
        })
        
        # 排序（按F1-Score降序）
        metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
        
        # 打印表格
        print(metrics_df.to_string(index=False))
        
        # 计算宏平均和加权平均
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        weighted_precision = (precision * support).sum() / support.sum()
        weighted_recall = (recall * support).sum() / support.sum()
        weighted_f1 = (f1 * support).sum() / support.sum()
        
        print(f"\n【宏平均 (Macro Average)】")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall:    {macro_recall:.4f}")
        print(f"  F1-Score:  {macro_f1:.4f}")
        
        print(f"\n【加权平均 (Weighted Average)】")
        print(f"  Precision: {weighted_precision:.4f}")
        print(f"  Recall:    {weighted_recall:.4f}")
        print(f"  F1-Score:  {weighted_f1:.4f}")
        
        # 保存CSV
        csv_path = self.output_dir / 'per_class_metrics.csv'
        metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 指标已保存到: {csv_path}")
        
        # 可视化
        self.plot_per_class_metrics(metrics_df)
        
        return metrics_df
    
    def plot_per_class_metrics(self, metrics_df):
        """可视化每类别指标"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision
        ax1 = axes[0, 0]
        bars = ax1.barh(metrics_df['中文名'], metrics_df['Precision'], color='steelblue')
        ax1.set_xlabel('Precision')
        ax1.set_title('各类别 Precision', fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Recall
        ax2 = axes[0, 1]
        bars = ax2.barh(metrics_df['中文名'], metrics_df['Recall'], color='coral')
        ax2.set_xlabel('Recall')
        ax2.set_title('各类别 Recall', fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # F1-Score
        ax3 = axes[1, 0]
        bars = ax3.barh(metrics_df['中文名'], metrics_df['F1-Score'], color='lightgreen')
        ax3.set_xlabel('F1-Score')
        ax3.set_title('各类别 F1-Score', fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 样本数
        ax4 = axes[1, 1]
        bars = ax4.barh(metrics_df['中文名'], metrics_df['样本数'], color='mediumpurple')
        ax4.set_xlabel('样本数')
        ax4.set_title('各类别样本数', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'per_class_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 指标可视化已保存到: {output_path}")
        plt.show()
    
    def analyze_confusion_pairs(self, cm, top_k=10):
        """分析最容易混淆的类别对"""
        print(f"\n{'='*60}")
        print(f"🔍 分析最容易混淆的类别对（Top {top_k}）")
        print(f"{'='*60}\n")
        
        confusion_pairs = []
        
        # 遍历混淆矩阵（排除对角线）
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j:  # 排除正确分类
                    count = cm[i, j]
                    if count > 0:
                        confusion_pairs.append({
                            '真实类别': self.class_names[i],
                            '真实类别(中文)': self.class_names_cn.get(self.class_names[i], self.class_names[i]),
                            '预测类别': self.class_names[j],
                            '预测类别(中文)': self.class_names_cn.get(self.class_names[j], self.class_names[j]),
                            '混淆数量': int(count),
                            '占该类比例': count / cm[i].sum()
                        })
        
        # 排序
        confusion_df = pd.DataFrame(confusion_pairs)
        confusion_df = confusion_df.sort_values('混淆数量', ascending=False).head(top_k)
        
        print(confusion_df.to_string(index=False))
        
        # 保存
        csv_path = self.output_dir / 'confusion_pairs.csv'
        confusion_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 混淆对分析已保存到: {csv_path}")
        
        return confusion_df
    
    def generate_classification_report(self, y_true, y_pred):
        """生成详细的分类报告"""
        print(f"\n{'='*60}")
        print(f"📋 生成分类报告")
        print(f"{'='*60}\n")
        
        # 使用sklearn的classification_report
        target_names = [self.class_names_cn.get(cls, cls) for cls in self.class_names]
        report_str = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4
        )
        
        print(report_str)
        
        # 保存到文件
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("植物幼苗分类 - 分类报告\n")
            f.write("="*60 + "\n\n")
            f.write(report_str)
        
        print(f"\n✅ 分类报告已保存到: {report_path}")


def main():
    """主函数"""
    print("🌱" * 30)
    print("植物幼苗分类 - 模型评估与混淆矩阵")
    print("🌱" * 30)
    
    # 配置
    model_path = './output/checkpoints/best_model.pth'
    data_dir = './data/split/val'
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"\n❌ 错误: 未找到模型文件 {model_path}")
        print("请先运行 3_ResNet101_训练.py 来训练模型")
        return
    
    if not Path(data_dir).exists():
        print(f"\n❌ 错误: 未找到验证数据 {data_dir}")
        print("请先运行 1_数据准备.py 来准备数据")
        return
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path, data_dir)
    
    # 预测
    y_pred, y_true, probs = evaluator.predict()
    
    # 混淆矩阵
    cm, cm_norm = evaluator.plot_confusion_matrix(y_true, y_pred)
    
    # 每类别指标
    metrics_df = evaluator.analyze_per_class_metrics(y_true, y_pred)
    
    # 混淆对分析
    confusion_pairs = evaluator.analyze_confusion_pairs(cm, top_k=10)
    
    # 分类报告
    evaluator.generate_classification_report(y_true, y_pred)
    
    print("\n" + "🎉" * 30)
    print("评估完成！")
    print("🎉" * 30)
    print(f"\n所有结果已保存到: {evaluator.output_dir}")
    print("\n下一步:")
    print("  1. 分析混淆矩阵，找出容易混淆的类别")
    print("  2. 运行 5_集成学习（可选）.py 来尝试模型融合")
    print("  3. 运行 6_Kaggle提交.py 来生成提交文件")


if __name__ == '__main__':
    main()
