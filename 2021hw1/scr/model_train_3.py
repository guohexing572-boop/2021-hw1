import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
import torch.optim as optim
from tools.hw1_dataset import COVIDRegressionDataset_3,TransformedSubset
from tools.hw1_model import hw1
from  tools.hw1_commom_tools import plot_loss_curves,get_trained_model_feature_importance

#训练结束后观察哪个feature更重要

# 设置基础路径和设备
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU

if __name__ == "__main__":
    # ============================ 配置参数 ============================
    # 数据集路径
    # 获取上级文件夹
    parent_dir = os.path.dirname(BASE_DIR)
    # 获取训练集和测试集的地址
    train_csv_path = os.path.join(parent_dir, 'covid.train.csv')
    test_csv_path = os.path.join(parent_dir, 'covid.test.csv')

    # 创建日志目录，以当前时间命名
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练超参数
    MAX_EPOCH = 182  # 总训练轮数
    BATCH_SIZE = 128  # 批大小
    LR = 0.001  # 初始学习率
    log_interval = 1  # 日志记录间隔
    val_interval = 1  # 验证间隔
    start_epoch = -1  # 起始epoch（用于断点续训）
    VAL_RATIO = 0.2  # 验证集比例

    # ============================ step 1/5 数据加载 ============================
    # 训练数据变换：包含数据增强
    def train_transform(feature):
        return torch.FloatTensor(feature)
    def valid_transform(feature):
        return torch.FloatTensor(feature)

    # 创建完整训练数据集
    full_train_data = COVIDRegressionDataset_3(
        path=train_csv_path,
        transform=None,  # 先不应用transform
        normalize=True
    )
    # 获取训练集的scaler
    feature_scaler, target_scaler = full_train_data.get_scalers()

    # 分割训练集和验证集
    train_size = int((1 - VAL_RATIO) * len(full_train_data))
    val_size = len(full_train_data) - train_size

    train_dataset, val_dataset = random_split(
        full_train_data, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )


    # 应用不同的transform
    train_data = TransformedSubset(train_dataset, transform=train_transform)
    valid_data = TransformedSubset(val_dataset, transform=valid_transform)

    print(f"总训练样本: {len(full_train_data)}")
    print(f"分割后训练集: {len(train_data)}")
    print(f"分割后验证集: {len(valid_data)}")

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # ============================ step 2/5 模型定义 ============================
    model = hw1(input_size=45,hidden_size=128,output_size=1)
    model.to(device)  # 将模型移动到设备（GPU/CPU）

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.MSELoss()  # 回归任务用MSE损失
    # ============================ step 4/5 优化器 ============================
    # 使用SGD优化器，带动量和权重衰减
    optimizer = optim.Adam(model.parameters(), lr=LR)  # 建议用Adam，学习率小一点
    # ============================ step 5/5 训练循环 ============================
    train_losses = []
    val_losses = []

    for epoch in range(MAX_EPOCH):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        # 当前代码：只在 val_interval 时记录验证损失
        if epoch % val_interval == 0:  # 这样 val_losses 长度会小于 train_losses
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            val_losses.append(avg_val_loss)  # 每个epoch都记录

            # 只在特定间隔打印
            if epoch % val_interval == 0:
                print(f'Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # 创建检查点字典
            checkpoint = {
                "model_state_dict": model.state_dict(),  # 模型参数
                "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
                "epoch": epoch,  # 当前epoch
            }

            # 保存检查点
            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
    #


    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
    picture_path = os.path.join(BASE_DIR, "..", "results", time_str,'loss_curves.png')
    plot_loss_curves(train_losses, val_losses, picture_path)


    # ============================ 计算特征重要性 ============================
    # 准备测试数据用于特征重要性分析
    print("\n=== 计算特征重要性 ===")

    # 收集验证集的所有数据
    val_features = []
    val_targets = []
    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            val_features.append(data.cpu().numpy())
            val_targets.append(target.cpu().numpy())

    # 合并所有batch的数据
    val_features = np.vstack(val_features)
    val_targets = np.concatenate(val_targets)

    print(f"验证集特征形状: {val_features.shape}")
    print(f"验证集目标形状: {val_targets.shape}")

    # 计算特征重要性
    importance_df = get_trained_model_feature_importance(
        model=model,
        test_features=val_features,
        test_targets=val_targets
    )

    print("特征重要性排序:")
    print(importance_df.head(20))  # 显示前20个最重要的特征

    # 保存特征重要性结果
    importance_csv_path = os.path.join(log_dir, "feature_importance.csv")
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"特征重要性已保存至: {importance_csv_path}")









