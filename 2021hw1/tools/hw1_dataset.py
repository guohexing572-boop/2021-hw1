# 将数据读取
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取上级文件夹
parent_dir = os.path.dirname(BASE_DIR)

# print(f"上级文件夹: {parent_dir}")
# print(BASE_DIR)

#获取训练集和测试集的地址
train_csv_path = os.path.join(parent_dir, 'covid.train.csv')
test_csv_path = os.path.join(parent_dir, 'covid.test.csv')


train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# #获取数据的长度
# print(train_df.shape)       #(2700,95)
# print(test_df.shape)        #(893,94)

# 读取一行数据
# print(train_df.iloc[0])

class COVIDRegressionDataset(Dataset):
    def __init__(self, path, transform=None, normalize=True,
                 feature_scaler=None, target_scaler=None):
        """
        COVID 回归数据集
        Args:
            path: 数据路径
            transform: 数据变换
            normalize: 是否标准化
            feature_scaler: 预训练的特征标准化器
            target_scaler: 预训练的目标标准化器
        """
        self.path = path
        self.transform = transform
        self.normalize = normalize

        # 读取数据
        self.df = pd.read_csv(self.path)

        # 分离特征和目标
        self.features = self.df.iloc[:, :-1].values.astype(np.float32)
        self.targets = self.df.iloc[:, -1].values.astype(np.float32)

        # 标准化处理
        if normalize:
            # 特征标准化
            if feature_scaler is None:
                self.feature_scaler = StandardScaler()
                self.features = self.feature_scaler.fit_transform(self.features)
            else:
                self.feature_scaler = feature_scaler
                self.features = self.feature_scaler.transform(self.features)

            # 目标变量标准化
            if target_scaler is None:
                self.target_scaler = StandardScaler()
                self.targets = self.target_scaler.fit_transform(
                    self.targets.reshape(-1, 1)
                ).flatten()
            else:
                self.target_scaler = target_scaler
                self.targets = self.target_scaler.transform(
                    self.targets.reshape(-1, 1)
                ).flatten()
        else:
            self.feature_scaler = None
            self.target_scaler = None

        print(f"数据集加载完成: {len(self)} 个样本")
        print(f"特征维度: {self.get_feature_dim()}")
        if normalize:
            print(f"目标变量范围: [{self.targets.min():.3f}, {self.targets.max():.3f}]")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].copy()
        target = self.targets[idx]

        # 数据增强
        if self.transform:
            feature = self.transform(feature)
        else:
            feature = torch.FloatTensor(feature)

        target = torch.FloatTensor([target])  # 回归任务保持 (1,) 形状

        return feature, target

    def get_feature_dim(self):
        """获取特征维度"""
        return self.features.shape[1]

    def get_scalers(self):
        """
        获取标准化器
        Returns:
            feature_scaler: 特征标准化器
            target_scaler: 目标标准化器
        """
        if not self.normalize:
            raise ValueError("数据集未进行标准化，无法获取标准化器")

        return self.feature_scaler, self.target_scaler

    def inverse_transform_target(self, targets):
        """
        将标准化后的目标值转换回原始尺度
        Args:
            targets: 标准化后的目标值
        Returns:
            原始尺度的目标值
        """
        if not self.normalize:
            return targets

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        return self.target_scaler.inverse_transform(
            targets.reshape(-1, 1)
        ).flatten()

    def inverse_transform_features(self, features):
        """
        将标准化后的特征值转换回原始尺度
        Args:
            features: 标准化后的特征值
        Returns:
            原始尺度的特征值
        """
        if not self.normalize:
            return features

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        return self.feature_scaler.inverse_transform(features)

# 为分割后的数据集分别设置transform
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x.numpy())  # 转回numpy应用transform
        return x, y

    def __len__(self):
        return len(self.subset)

if __name__ == "__main__":
    train_data = COVIDRegressionDataset(train_csv_path)
    print(train_data.targets)