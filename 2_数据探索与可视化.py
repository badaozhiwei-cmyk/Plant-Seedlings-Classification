"""
🌱 植物幼苗分类 - 数据探索与可视化
===================================
功能：
1. 探索性数据分析（EDA）
2. 可视化样本图像
3. 分析图像尺寸分布
4. 检查图像质量
5. 数据增强示例

作者：霸道志伟
日期：2026-02-15
"""

import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PlantSeedlingsEDA:
    """植物幼苗数据探索分析类"""
    
    def __init__(self, data_dir='./data/train'):
        """
        初始化EDA分析器
        
        Args:
            data_dir: 训练数据目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path('./data/eda_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 12个植物类别（中英文对照）
        self.class_names = {
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
        
    def visualize_sample_images(self, samples_per_class=3):
        """
        可视化每个类别的样本图像
        
        Args:
            samples_per_class: 每个类别显示的样本数
        """
        print("=" * 60)
        print("🖼️ 可视化样本图像")
        print("=" * 60)
        
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        num_classes = len(classes)
        
        # 创建子图
        fig, axes = plt.subplots(num_classes, samples_per_class,
                                 figsize=(samples_per_class * 3, num_classes * 3))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, cls in enumerate(classes):
            cls_dir = self.data_dir / cls
            images = list(cls_dir.glob('*.png'))
            
            # 随机选择样本
            samples = random.sample(images, min(samples_per_class, len(images)))
            
            for j, img_path in enumerate(samples):
                img = Image.open(img_path)
                
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                
                # 只在第一列显示类别名
                if j == 0:
                    chinese_name = self.class_names.get(cls, cls)
                    axes[i, j].set_title(f'{chinese_name}\n{cls}',
                                        fontsize=10, loc='left')
                
                # 显示图像尺寸
                axes[i, j].text(0.5, -0.1, f'{img.size[0]}x{img.size[1]}',
                               transform=axes[i, j].transAxes,
                               ha='center', fontsize=8, color='gray')
        
        plt.suptitle('各类别样本图像', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'sample_images.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✅ 样本图像已保存到: {output_path}")
        plt.show()
        
    def analyze_image_sizes(self):
        """分析图像尺寸分布"""
        print("\n" + "=" * 60)
        print("📏 分析图像尺寸分布")
        print("=" * 60)
        
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        all_sizes = []
        widths = []
        heights = []
        aspects = []
        
        print("\n正在扫描图像...")
        for cls in tqdm(classes, desc="处理类别"):
            cls_dir = self.data_dir / cls
            for img_path in cls_dir.glob('*.png'):
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    
                    widths.append(w)
                    heights.append(h)
                    all_sizes.append((w, h))
                    aspects.append(w / h)
                except Exception as e:
                    print(f"警告: 无法读取 {img_path}: {e}")
        
        # 统计信息
        print(f"\n【图像尺寸统计】")
        print(f"  总图像数: {len(all_sizes)}")
        print(f"  宽度范围: {min(widths)} - {max(widths)} (平均: {np.mean(widths):.1f})")
        print(f"  高度范围: {min(heights)} - {max(heights)} (平均: {np.mean(heights):.1f})")
        print(f"  宽高比范围: {min(aspects):.2f} - {max(aspects):.2f} (平均: {np.mean(aspects):.2f})")
        
        # 统计最常见的尺寸
        size_counter = Counter(all_sizes)
        print(f"\n【最常见的尺寸（Top 5）】")
        for size, count in size_counter.most_common(5):
            print(f"  {size[0]:4d} x {size[1]:4d}: {count:4d} 张 ({count/len(all_sizes)*100:.1f}%)")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 宽度分布
        axes[0, 0].hist(widths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--',
                          label=f'平均: {np.mean(widths):.1f}')
        axes[0, 0].set_xlabel('宽度 (像素)')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('图像宽度分布')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 高度分布
        axes[0, 1].hist(heights, bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--',
                          label=f'平均: {np.mean(heights):.1f}')
        axes[0, 1].set_xlabel('高度 (像素)')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('图像高度分布')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 宽高比分布
        axes[1, 0].hist(aspects, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(aspects), color='red', linestyle='--',
                          label=f'平均: {np.mean(aspects):.2f}')
        axes[1, 0].set_xlabel('宽高比')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('宽高比分布')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 散点图：宽度 vs 高度
        axes[1, 1].scatter(widths, heights, alpha=0.3, s=5)
        axes[1, 1].set_xlabel('宽度 (像素)')
        axes[1, 1].set_ylabel('高度 (像素)')
        axes[1, 1].set_title('宽度 vs 高度')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'image_size_analysis.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✅ 尺寸分析图已保存到: {output_path}")
        plt.show()
        
        return {
            'widths': widths,
            'heights': heights,
            'aspects': aspects,
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights)
        }
    
    def analyze_color_distribution(self, num_samples=500):
        """
        分析颜色分布
        
        Args:
            num_samples: 采样图像数量
        """
        print("\n" + "=" * 60)
        print("🎨 分析颜色分布")
        print("=" * 60)
        
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        # 收集所有图像路径
        all_images = []
        for cls in classes:
            cls_dir = self.data_dir / cls
            all_images.extend(list(cls_dir.glob('*.png')))
        
        # 随机采样
        sampled_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        # 收集颜色统计
        r_values = []
        g_values = []
        b_values = []
        
        print(f"\n正在分析 {len(sampled_images)} 张图像的颜色...")
        for img_path in tqdm(sampled_images, desc="处理图像"):
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 采样像素（每张图随机采样1000个像素）
                h, w = img.shape[:2]
                num_pixels = min(1000, h * w)
                indices = random.sample(range(h * w), num_pixels)
                
                flat_img = img.reshape(-1, 3)
                sampled_pixels = flat_img[indices]
                
                r_values.extend(sampled_pixels[:, 0])
                g_values.extend(sampled_pixels[:, 1])
                b_values.extend(sampled_pixels[:, 2])
                
            except Exception as e:
                print(f"警告: 无法读取 {img_path}: {e}")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RGB分布
        ax1 = axes[0, 0]
        ax1.hist(r_values, bins=50, color='red', alpha=0.5, label='R', density=True)
        ax1.hist(g_values, bins=50, color='green', alpha=0.5, label='G', density=True)
        ax1.hist(b_values, bins=50, color='blue', alpha=0.5, label='B', density=True)
        ax1.set_xlabel('像素值')
        ax1.set_ylabel('密度')
        ax1.set_title('RGB通道分布')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # R通道
        axes[0, 1].hist(r_values, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('R值')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('红色通道分布')
        axes[0, 1].grid(alpha=0.3)
        
        # G通道
        axes[1, 0].hist(g_values, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('G值')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('绿色通道分布')
        axes[1, 0].grid(alpha=0.3)
        
        # B通道
        axes[1, 1].hist(b_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('B值')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('蓝色通道分布')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'color_distribution.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✅ 颜色分布图已保存到: {output_path}")
        plt.show()
        
        print(f"\n【颜色统计】")
        print(f"  R通道: 均值={np.mean(r_values):.1f}, 标准差={np.std(r_values):.1f}")
        print(f"  G通道: 均值={np.mean(g_values):.1f}, 标准差={np.std(g_values):.1f}")
        print(f"  B通道: 均值={np.mean(b_values):.1f}, 标准差={np.std(b_values):.1f}")
        
    def show_data_augmentation_examples(self):
        """展示数据增强示例"""
        print("\n" + "=" * 60)
        print("🔄 数据增强示例")
        print("=" * 60)
        
        # 随机选择一张图像
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        random_class = random.choice(classes)
        cls_dir = self.data_dir / random_class
        images = list(cls_dir.glob('*.png'))
        sample_img_path = random.choice(images)
        
        # 读取图像
        img = cv2.imread(str(sample_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 定义增强操作
        augmentations = {
            '原始图像': img,
            '水平翻转': cv2.flip(img, 1),
            '垂直翻转': cv2.flip(img, 0),
            '旋转90°': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            '旋转180°': cv2.rotate(img, cv2.ROTATE_180),
            '亮度增强': cv2.convertScaleAbs(img, alpha=1.3, beta=30),
            '亮度降低': cv2.convertScaleAbs(img, alpha=0.7, beta=-30),
            '模糊': cv2.GaussianBlur(img, (5, 5), 0),
        }
        
        # 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for idx, (name, aug_img) in enumerate(augmentations.items()):
            axes[idx].imshow(aug_img)
            axes[idx].set_title(name, fontsize=12)
            axes[idx].axis('off')
        
        chinese_name = self.class_names.get(random_class, random_class)
        plt.suptitle(f'数据增强示例 - {chinese_name} ({random_class})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'augmentation_examples.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✅ 数据增强示例已保存到: {output_path}")
        plt.show()
        
        print("\n💡 建议的数据增强策略：")
        print("  1. 随机旋转（0-360°）")
        print("  2. 随机翻转（水平/垂直）")
        print("  3. 随机缩放（0.8-1.2倍）")
        print("  4. 颜色抖动（亮度、对比度、饱和度）")
        print("  5. 随机裁剪")
        print("  6. 高斯模糊（适度）")


def main():
    """主函数"""
    print("🌱" * 30)
    print("植物幼苗分类 - 数据探索与可视化")
    print("🌱" * 30)
    
    # 创建EDA分析器
    eda = PlantSeedlingsEDA('./data/train')
    
    # 确保数据目录存在
    if not eda.data_dir.exists():
        print(f"\n❌ 错误: 找不到数据目录 {eda.data_dir}")
        print("请先运行 1_数据准备.py 来下载和准备数据")
        return
    
    print("\n开始探索性数据分析...\n")
    
    # 1. 可视化样本图像
    eda.visualize_sample_images(samples_per_class=3)
    
    # 2. 分析图像尺寸
    size_info = eda.analyze_image_sizes()
    
    # 3. 分析颜色分布
    eda.analyze_color_distribution(num_samples=500)
    
    # 4. 展示数据增强示例
    eda.show_data_augmentation_examples()
    
    # 总结建议
    print("\n" + "=" * 60)
    print("📝 EDA总结与建议")
    print("=" * 60)
    
    print(f"\n【数据预处理建议】")
    print(f"  1. 统一图像尺寸: 推荐 224x224 或 256x256")
    print(f"     - 当前平均尺寸: {size_info['mean_width']:.0f}x{size_info['mean_height']:.0f}")
    print(f"  2. 数据标准化: 使用ImageNet的均值和标准差")
    print(f"     - mean=[0.485, 0.456, 0.406]")
    print(f"     - std=[0.229, 0.224, 0.225]")
    print(f"  3. 数据增强: 参考上面的增强示例")
    
    print(f"\n【模型训练建议】")
    print(f"  1. 使用预训练的ResNet-101（ImageNet权重）")
    print(f"  2. 迁移学习: 冻结前几层，只训练后面的层")
    print(f"  3. 学习率: 从1e-4开始，使用学习率调度")
    print(f"  4. Batch size: 根据GPU内存，建议16-32")
    print(f"  5. Epochs: 20-50个epoch，使用早停")
    
    print("\n" + "🎉" * 30)
    print("EDA完成！")
    print("🎉" * 30)
    print("\n下一步: 运行 3_ResNet101_训练.py")


if __name__ == '__main__':
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    main()
