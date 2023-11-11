# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:59:53 2023

@author: luzy1
"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐状态
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        # 全连接层
        out = self.fc(out)
        
        return out



    def fea(self,x):
        
        # x = x.unsqueeze(1)
        # 初始化隐状态
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # x = x.view(x.size(0), -1)
        
        return out
        
    
    
































'''


# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:38:56 2023

@author: luzy1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# 定义超参数
input_size = 1 # 输入信号的维度
hidden_size = 64 # LSTM的隐藏层大小
num_layers = 2 # LSTM的层数
num_classes = 4 # 分类的类别数
sequence_length = 1200 # 输入信号的长度
batch_size = 32 # 批次大小
num_epochs = 10 # 训练的轮数
learning_rate = 0.01 # 学习率

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # LSTM层
        self.fc = nn.Linear(hidden_size, num_classes) # 全连接层

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层
        out = self.fc(out)

        return out

# 创建模拟数据集，您可以用您自己的数据集替换
X = torch.randn(1000, sequence_length, input_size) # 一维信号数据，形状为(样本数，序列长度，输入维度)
y = torch.randint(0, num_classes, (1000,)) # 标签数据，形状为(样本数，)

# 划分训练集和测试集
train_size = int(0.8 * len(X)) # 训练集大小，这里取80%的数据作为训练集
test_size = len(X) - train_size # 测试集大小，这里取20%的数据作为测试集
train_X, test_X = torch.utils.data.random_split(X, [train_size, test_size]) # 随机划分训练集和测试集的一维信号数据
train_y, test_y = torch.utils.data.random_split(y, [train_size, test_size]) # 随机划分训练集和测试集的标签数据

# 创建数据加载器
train_dataset = data.TensorDataset(train_X, train_y) # 创建训练集的数据集对象
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 创建训练集的数据加载器对象，可以批量加载数据并打乱顺序
test_dataset = data.TensorDataset(test_X, test_y) # 创建测试集的数据集对象
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 创建测试集的数据加载器对象，可以批量加载数据

# 检测是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型对象
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

# 训练模型
total_step = len(train_loader) # 训练集的总批次数
for epoch in range(num_epochs): # 对于每一轮训练
    for i, (signals, labels) in enumerate(train_loader): # 对于每一个批次的数据
        signals = signals.reshape(-1, sequence_length, input_size).to(device) # 将一维信号数据转换为模型需要的形状并发送到设备上
        labels = labels.to(device) # 将标签数据发送到设备上

        # 前向传播
        outputs = model(signals) # 通过模型得到输出
        loss = criterion(outputs, labels) # 计算损失

        # 反向传播和优化
        optimizer.zero_grad() # 清零梯度
        loss.backward() # 反向计算梯度
        optimizer.step() # 更新参数

        # 打印训练信息
        if (i + 1) % 10 == 0: # 每10个批次打印一次
            print(f'Epoch {epoch + 1}, Step {i + 1}, Loss {loss.item():.4f}')

# 测试模型
with torch.no_grad(): # 不计算梯度，节省内存
    correct = 0 # 预测正确的样本数
    total = 0 # 总样本数
    for signals, labels in test_loader: # 对于测试集中的每一个批次的数据
        signals = signals.reshape(-1, sequence_length, input_size).to(device) # 将一维信号数据转换为模型需要的形状并发送到设备上
        labels = labels.to(device) # 将标签数据发送到设备上
        outputs = model(signals) # 通过模型得到输出
        _, predicted = torch.max(outputs.data, 1) # 得到预测的类别
        total += labels.size(0) # 更新总样本数
        correct += (predicted == labels).sum().item() # 更新预测正确的样本数

    # 打印测试结果
    print(f'测试集的准确率为 {100 * correct / total:.2f}%')
'''