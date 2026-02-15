"""
🌱 植物幼苗分类 - 数据准备
===================================
功能：
1. 下载Kaggle数据集
2. 数据集结构分析
3. 数据集划分（训练集/验证集）
4. 统计类别分布（检查数据不平衡）

作者：霸道志伟
日期：2026-02-15
"""

import os
import shutil
import json
from pathlib import Path
from collections import Counter
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class PlantSeedlingsDataPreparation:
    """植物幼苗数据准备类"""
    
    def __init__(self, base_dir='./data'):
        """
        初始化数据准备器
        
        Args:
            base_dir: 数据存储的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据路径
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        self.split_dir = self.base_dir / 'split'
        
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
        
    def download_data_from_kaggle(self):
        """
        从Kaggle下载数据集
        
        前置条件：
        1. 安装 kaggle api: pip install kaggle
        2. 配置 kaggle.json 到 ~/.kaggle/ 目录
        3. 接受竞赛规则: https://www.kaggle.com/c/plant-seedlings-classification/rules
        """
        print("=" * 60)
        print("📥 从Kaggle下载数据集")
        print("=" * 60)
        
        try:
            import kaggle
            
            # 下载数据集
            print("\n正在下载数据集...")
            kaggle.api.competition_download_files(
                'plant-seedlings-classification',
                path=str(self.base_dir)
            )
            
            # 解压数据
            zip_file = self.base_dir / 'plant-seedlings-classification.zip'
            if zip_file.exists():
                print(f"\n正在解压 {zip_file}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)
                print("✅ 解压完成！")
                
                # 删除zip文件
                zip_file.unlink()
            
            print("\n✅ 数据下载完成！")
            
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("\n💡 请按以下步骤操作：")
            print("1. 安装 Kaggle API: pip install kaggle")
            print("2. 从 https://www.kaggle.com/settings 下载 kaggle.json")
            print("3. 将 kaggle.json 放到 ~/.kaggle/ 目录")
            print("4. 在 https://www.kaggle.com/c/plant-seedlings-classification 接受竞赛规则")
            print("\n或者手动下载数据集到 ./data 目录")
            
    def analyze_dataset_structure(self):
        """分析数据集结构"""
        print("\n" + "=" * 60)
        print("📊 数据集结构分析")
        print("=" * 60)
        
        # 训练集结构
        if self.train_dir.exists():
            print("\n【训练集结构】")
            classes = sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])
            class_counts = {}
            
            for cls in classes:
                cls_dir = self.train_dir / cls
                num_images = len(list(cls_dir.glob('*.png')))
                class_counts[cls] = num_images
                chinese_name = self.class_names.get(cls, cls)
                print(f"  {cls:30s} ({chinese_name:12s}): {num_images:4d} 张图像")
            
            total_images = sum(class_counts.values())
            print(f"\n  总计: {len(classes)} 个类别, {total_images} 张图像")
            
            # 返回类别计数用于后续分析
            return class_counts
        else:
            print("\n⚠️ 训练集目录不存在！")
            return {}
    
    def visualize_class_distribution(self, class_counts):
        """
        可视化类别分布
        
        Args:
            class_counts: 类别计数字典
        """
        print("\n" + "=" * 60)
        print("📈 可视化类别分布")
        print("=" * 60)
        
        if not class_counts:
            print("⚠️ 没有类别数据！")
            return
        
        # 创建DataFrame
        df = pd.DataFrame([
            {
                '类别': cls,
                '中文名': self.class_names.get(cls, cls),
                '图像数量': count
            }
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        # 计算统计信息
        total = df['图像数量'].sum()
        mean_count = df['图像数量'].mean()
        max_count = df['图像数量'].max()
        min_count = df['图像数量'].min()
        imbalance_ratio = max_count / min_count
        
        print(f"\n【数据不平衡分析】")
        print(f"  总图像数: {total}")
        print(f"  平均每类: {mean_count:.1f}")
        print(f"  最多的类: {max_count}")
        print(f"  最少的类: {min_count}")
        print(f"  不平衡比: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print(f"\n  ⚠️ 检测到数据不平衡（比例 {imbalance_ratio:.2f}:1）")
            print(f"  💡 建议使用：")
            print(f"     1. 加权损失函数")
            print(f"     2. 过采样/欠采样")
            print(f"     3. Focal Loss")
        else:
            print(f"\n  ✅ 数据相对平衡")
        
        # 绘制条形图
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图1: 按类别排序
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(df)), df['图像数量'], color='steelblue', alpha=0.8)
        ax1.axhline(y=mean_count, color='red', linestyle='--', label=f'平均值: {mean_count:.1f}')
        ax1.set_xlabel('类别', fontsize=12)
        ax1.set_ylabel('图像数量', fontsize=12)
        ax1.set_title('各类别图像数量分布（按数量排序）', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['中文名'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 在每个柱子上显示数值
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        # 图2: 百分比饼图
        ax2 = axes[1]
        colors = plt.cm.Set3(range(len(df)))
        wedges, texts, autotexts = ax2.pie(
            df['图像数量'],
            labels=df['中文名'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax2.set_title('各类别占比', fontsize=14, fontweight='bold')
        
        # 设置文字大小
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.base_dir / 'class_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 类别分布图已保存到: {output_path}")
        plt.show()
        
        # 保存统计CSV
        csv_path = self.base_dir / 'class_statistics.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 类别统计已保存到: {csv_path}")
        
        return df
    
    def split_train_validation(self, val_ratio=0.2, random_state=42):
        """
        划分训练集和验证集
        
        Args:
            val_ratio: 验证集比例
            random_state: 随机种子
        """
        print("\n" + "=" * 60)
        print(f"✂️ 划分训练集和验证集 (验证集比例: {val_ratio * 100}%)")
        print("=" * 60)
        
        # 创建输出目录
        train_split_dir = self.split_dir / 'train'
        val_split_dir = self.split_dir / 'val'
        
        train_split_dir.mkdir(parents=True, exist_ok=True)
        val_split_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有类别
        classes = sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])
        
        split_info = []
        
        for cls in classes:
            # 创建类别目录
            (train_split_dir / cls).mkdir(exist_ok=True)
            (val_split_dir / cls).mkdir(exist_ok=True)
            
            # 获取该类别的所有图像
            cls_dir = self.train_dir / cls
            images = list(cls_dir.glob('*.png'))
            
            # 划分
            train_images, val_images = train_test_split(
                images,
                test_size=val_ratio,
                random_state=random_state
            )
            
            # 复制文件
            for img in train_images:
                shutil.copy2(img, train_split_dir / cls / img.name)
            
            for img in val_images:
                shutil.copy2(img, val_split_dir / cls / img.name)
            
            chinese_name = self.class_names.get(cls, cls)
            print(f"  {cls:30s} ({chinese_name:12s}): "
                  f"训练 {len(train_images):4d} | 验证 {len(val_images):4d}")
            
            split_info.append({
                '类别': cls,
                '中文名': chinese_name,
                '总数': len(images),
                '训练集': len(train_images),
                '验证集': len(val_images)
            })
        
        # 保存划分信息
        split_df = pd.DataFrame(split_info)
        split_csv = self.split_dir / 'split_info.csv'
        split_df.to_csv(split_csv, index=False, encoding='utf-8-sig')
        
        total_train = split_df['训练集'].sum()
        total_val = split_df['验证集'].sum()
        
        print(f"\n【划分结果】")
        print(f"  训练集: {total_train} 张图像")
        print(f"  验证集: {total_val} 张图像")
        print(f"  总计: {total_train + total_val} 张图像")
        print(f"\n✅ 数据划分完成！")
        print(f"✅ 划分信息已保存到: {split_csv}")
        
        return split_df
    
    def calculate_class_weights(self, class_counts):
        """
        计算类别权重（用于加权损失函数）
        
        Args:
            class_counts: 类别计数字典
            
        Returns:
            类别权重字典
        """
        print("\n" + "=" * 60)
        print("⚖️ 计算类别权重")
        print("=" * 60)
        
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # 计算权重: weight = total / (num_classes * class_count)
        # 这样少数类会得到更高的权重
        class_weights = {}
        
        print(f"\n{'类别':<30s} {'图像数':<10s} {'权重':<10s}")
        print("-" * 60)
        
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
            weight = total / (num_classes * count)
            class_weights[cls] = weight
            chinese_name = self.class_names.get(cls, cls)
            print(f"{cls:<30s} {count:<10d} {weight:<10.4f} ({chinese_name})")
        
        # 保存权重
        weights_file = self.base_dir / 'class_weights.json'
        with open(weights_file, 'w', encoding='utf-8') as f:
            json.dump(class_weights, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 类别权重已保存到: {weights_file}")
        print(f"💡 在训练时使用这些权重可以缓解数据不平衡问题")
        
        return class_weights


def main():
    """主函数"""
    print("🌱" * 30)
    print("植物幼苗分类 - 数据准备")
    print("🌱" * 30)
    
    # 创建数据准备器
    prep = PlantSeedlingsDataPreparation('./data')
    
    # 步骤1: 下载数据（如果需要）
    print("\n" + "=" * 60)
    print("步骤1: 数据下载")
    print("=" * 60)
    download = input("\n是否需要从Kaggle下载数据？(y/n): ").strip().lower()
    if download == 'y':
        prep.download_data_from_kaggle()
    else:
        print("跳过下载步骤，请确保数据已在 ./data 目录中")
    
    # 步骤2: 分析数据集结构
    class_counts = prep.analyze_dataset_structure()
    
    if not class_counts:
        print("\n❌ 未找到训练数据，程序退出")
        return
    
    # 步骤3: 可视化类别分布
    prep.visualize_class_distribution(class_counts)
    
    # 步骤4: 计算类别权重
    class_weights = prep.calculate_class_weights(class_counts)
    
    # 步骤5: 划分训练集和验证集
    split_df = prep.split_train_validation(val_ratio=0.2, random_state=42)
    
    print("\n" + "🎉" * 30)
    print("数据准备完成！")
    print("🎉" * 30)
    print("\n下一步: 运行 2_数据探索与可视化.py")
    

if __name__ == '__main__':
    main()
