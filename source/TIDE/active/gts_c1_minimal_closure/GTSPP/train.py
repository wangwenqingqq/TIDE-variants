import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ================= 配置 =================
# 你的 CSV 文件路径
csv_file = "training_data_sift.csv" 
# 剪枝激进程度 (0.0 ~ 1.0)
# 0.9 表示: 只有当下界距离 <= 当前半径的 90% 时才保留。
# 这意味着我们教 AI 剪掉那些"处于边缘" (90%~100%) 的节点。
PRUNING_THRESHOLD = 0.86 
# =======================================

# 1. 加载数据
print(f"Loading {csv_file}...")
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} samples.")
print("Sample raw data:\n", df.head())

# 2. 标签生成 (Label Generation) - 这一步最关键！
# 你的原始数据里没有 label，我们现在人工制造一个"更聪明"的策略。
# 策略：如果 lb 确实很小 (<= r * 0.9)，标记为 1 (保留)；
#       如果 lb 比较大 (接近 r)，虽然数学上能过，但我们教 AI 把它剪掉，标记为 0 (剪枝)。
# 注意：我们这里用 'node_dist' 对应你 CSV 里的列名
df['label'] = df.apply(lambda row: 1 if row['lb'] <= row['r'] * PRUNING_THRESHOLD else 0, axis=1)

# 打印一下正负样本比例，防止数据倾斜太严重
pos_count = len(df[df['label'] == 1])
neg_count = len(df[df['label'] == 0])
print(f"Keep (Label=1): {pos_count}, Prune (Label=0): {neg_count}")

# 3. 准备训练数据
# 提取特征列: lb, r, node_dist
X_raw = df[['lb', 'r', 'node_dist']].values.astype(np.float32)
y_raw = df['label'].values.astype(np.float32).reshape(-1, 1)

# 归一化 (Normalization)
# 获取每一列的最大值，用于将数据缩放到 [0, 1] 区间
# 加上 1e-6 是为了防止最大值为 0 导致除以 0 错误
X_max = np.max(X_raw, axis=0) + 1e-6
X_data = X_raw / X_max

# 转为 PyTorch Tensor
X_tensor = torch.from_numpy(X_data)
y_tensor = torch.from_numpy(y_raw)

# 4. 定义极简模型 (3输入 -> 8隐藏 -> 1输出)
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 8)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = TinyMLP()
criterion = nn.BCELoss() # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 开始训练
print("\nStarting training...")
for epoch in range(1000): # 训练 1000 轮
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        accuracy = ((outputs > 0.5).float() == y_tensor).float().mean()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {accuracy.item():.4f}")

# 6. 导出 C++ 代码
print("\n" + "="*40)
print("   Exporting weights to weight.txt")
print("="*40 + "\n")

with open("weight_gist.txt", "w") as f:
    # 导出归一化系数
    f.write("// Normalization factors (1 / max_value)\n")
    f.write(f"__constant__ float input_scale[3] = {{{1.0/X_max[0]:.8f}, {1.0/X_max[1]:.8f}, {1.0/X_max[2]:.8f}}};\n")

    # 导出 Layer 1 权重
    f.write("\n// Layer 1 Weights (8x3)\n")
    f.write("__constant__ float w1[8][3] = {\n")
    w1 = model.fc1.weight.detach().numpy()
    for row in w1:
        f.write(f"    {{{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}}},\n")
    f.write("};\n")

    # 导出 Layer 1 偏置
    f.write("\n// Layer 1 Bias\n")
    f.write("__constant__ float b1[8] = {\n")
    b1 = model.fc1.bias.detach().numpy()
    f.write("    " + ", ".join([f"{x:.6f}" for x in b1]) + "\n")
    f.write("};\n")

    # 导出 Layer 2 权重 (简化为一维数组)
    f.write("\n// Layer 2 Weights (1x8)\n")
    f.write("__constant__ float w2[8] = {\n")
    w2 = model.fc2.weight.detach().numpy().flatten()
    f.write("    " + ", ".join([f"{x:.6f}" for x in w2]) + "\n")
    f.write("};\n")

    # 导出 Layer 2 偏置
    f.write("\n// Layer 2 Bias\n")
    f.write(f"__constant__ float b2 = {model.fc2.bias.item():.6f};\n")

print("Weights exported to weight_gist.txt")