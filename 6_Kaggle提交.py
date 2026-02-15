"""
🌱 植物幼苗分类 - Kaggle提交
===================================
功能：
1. 加载训练好的模型
2. 对测试集进行预测
3. 生成符合Kaggle要求的提交文件

作者：霸道志伟
日期：2026-02-15
"""

import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm


class PlantSeedlingsTestDataset(Dataset):
    """测试集数据集（无标签）"""
    
    def __init__(self, test_dir, transform=None):
        """
        Args:
            test_dir: 测试集目录
            transform: 数据转换
        """
        self.test_dir = Path(test_dir)
        self.transform = transform
        
        # 收集所有测试图像
        self.image_paths = sorted(list(self.test_dir.glob('*.png')))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和文件名（用于生成提交文件）
        filename = img_path.name
        
        return image, filename


class KaggleSubmissionGenerator:
    """Kaggle提交文件生成器"""
    
    def __init__(self, model_path, test_dir, output_path, device=None):
        """
        Args:
            model_path: 模型checkpoint路径
            test_dir: 测试集目录
            output_path: 提交文件输出路径
            device: 计算设备
        """
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        self.output_path = Path(output_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 类别名称
        self.class_names = [
            'Black-grass',
            'Charlock',
            'Cleavers',
            'Common Chickweed',
            'Common wheat',
            'Fat Hen',
            'Loose Silky-bent',
            'Maize',
            'Scentless Mayweed',
            'Shepherds Purse',
            'Small-flowered Cranesbill',
            'Sugar beet'
        ]
        
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
        
        # 重建模型
        model = models.resnet101(weights=None)
        num_features = model.fc.in_features
        num_classes = len(self.class_names)
        model.fc = nn.Linear(num_features, num_classes)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ 模型加载成功！")
        
        return model
    
    def load_test_data(self):
        """加载测试数据"""
        print(f"\n{'='*60}")
        print(f"📁 加载测试数据")
        print(f"{'='*60}")
        
        # 数据转换（与训练时的验证集相同）
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 创建数据集
        dataset = PlantSeedlingsTestDataset(self.test_dir, transform=transform)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        print(f"  测试集目录: {self.test_dir}")
        print(f"  测试样本数: {len(dataset)}")
        
        return dataloader
    
    def predict_test_set(self, model, dataloader):
        """对测试集进行预测"""
        print(f"\n{'='*60}")
        print(f"🔮 预测测试集")
        print(f"{'='*60}\n")
        
        predictions = []
        
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc="预测中"):
                images = images.to(self.device)
                
                # 前向传播
                outputs = model(images)
                _, preds = outputs.max(1)
                
                # 收集预测结果
                for filename, pred in zip(filenames, preds):
                    pred_class = self.class_names[pred.item()]
                    predictions.append({
                        'file': filename,
                        'species': pred_class
                    })
        
        print(f"✅ 预测完成！总共 {len(predictions)} 个测试样本")
        
        return predictions
    
    def generate_submission_file(self, predictions):
        """生成Kaggle提交文件"""
        print(f"\n{'='*60}")
        print(f"📄 生成提交文件")
        print(f"{'='*60}")
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入CSV文件
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'species'])
            writer.writeheader()
            writer.writerows(predictions)
        
        print(f"✅ 提交文件已生成: {self.output_path}")
        print(f"  文件格式: CSV")
        print(f"  行数: {len(predictions) + 1} (含表头)")
        
        # 显示前几行
        print(f"\n【预览前5行】")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"  {i}. {pred['file']:<30s} -> {pred['species']}")
        
        return self.output_path
    
    def run(self):
        """运行完整的预测和提交流程"""
        # 加载模型
        model = self.load_model()
        
        # 加载测试数据
        dataloader = self.load_test_data()
        
        # 预测
        predictions = self.predict_test_set(model, dataloader)
        
        # 生成提交文件
        submission_path = self.generate_submission_file(predictions)
        
        return submission_path


def main():
    """主函数"""
    print("🌱" * 30)
    print("植物幼苗分类 - Kaggle提交")
    print("🌱" * 30)
    
    # 配置
    model_path = './output/checkpoints/best_model.pth'
    test_dir = './data/test'
    output_path = './output/submission.csv'
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"\n❌ 错误: 未找到模型文件 {model_path}")
        print("请先运行 3_ResNet101_训练.py 来训练模型")
        return
    
    if not Path(test_dir).exists():
        print(f"\n❌ 错误: 未找到测试数据 {test_dir}")
        print("请先运行 1_数据准备.py 来下载数据")
        return
    
    # 创建提交生成器
    generator = KaggleSubmissionGenerator(
        model_path=model_path,
        test_dir=test_dir,
        output_path=output_path
    )
    
    # 运行
    submission_path = generator.run()
    
    # 完成
    print("\n" + "🎉" * 30)
    print("提交文件生成完成！")
    print("🎉" * 30)
    
    print(f"\n【下一步】")
    print(f"1. 前往 Kaggle 竞赛页面:")
    print(f"   https://www.kaggle.com/c/plant-seedlings-classification")
    print(f"2. 上传提交文件: {submission_path}")
    print(f"3. 查看你的分数和排名")
    
    print(f"\n💡 提升建议：")
    print(f"  1. 尝试数据增强的不同组合")
    print(f"  2. 调整学习率和训练轮次")
    print(f"  3. 尝试不同的损失函数（Focal Loss）")
    print(f"  4. 使用模型融合（运行 5_集成学习（可选）.py）")
    print(f"  5. 尝试其他预训练模型（EfficientNet, Vision Transformer等）")


if __name__ == '__main__':
    main()
