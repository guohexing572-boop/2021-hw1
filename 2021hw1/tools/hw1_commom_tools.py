import numpy as np
import torch
import sys
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models
sys.path.append('../tools')
import pandas as pd


def plot_loss_curves(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")


def get_trained_model_feature_importance(model, test_features, test_targets):
    """
    计算已训练模型的特征重要性（基于梯度方法）

    Args:
        model: 已经训练好的PyTorch模型
        test_features: 测试集特征
        test_targets: 测试集目标

    Returns:
        importance_df: 包含所有特征重要性的DataFrame
    """
    model.eval()

    # 转换为tensor
    features_tensor = torch.FloatTensor(test_features).requires_grad_(True)
    targets_tensor = torch.FloatTensor(test_targets)

    # 前向传播
    outputs = model(features_tensor)
    loss = torch.nn.MSELoss()(outputs.squeeze(), targets_tensor)

    # 反向传播计算梯度
    model.zero_grad()
    loss.backward()

    # 计算特征重要性（梯度绝对值均值）
    gradients = torch.abs(features_tensor.grad)
    feature_importance = gradients.mean(dim=0).detach().numpy()

    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature_index': range(len(feature_importance)),
        'feature_name': [f'feature_{i}' for i in range(len(feature_importance))],
        'importance': feature_importance,
        'importance_normalized': feature_importance / feature_importance.sum()
    })

    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df

