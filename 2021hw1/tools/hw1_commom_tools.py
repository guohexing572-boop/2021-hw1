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