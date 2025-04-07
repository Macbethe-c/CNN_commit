import torch
import torch.utils
import torchvision
import torchvision.transforms as transforms
from torch import nn
alpha = 0.1
# 定义数据预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 PyTorch 的 Tensor 格式
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
])

# this is a test
# 打印图片的 shape 和对应的类别标签
# print(f'Image batch dimensions: {images.shape}')
# print(f'Label batch dimensions: {labels.shape}')
# print(f'First label: {classes[labels[0]]}')
from typing import Optional, Union
from torch.nn import functional as F
class CIFAR10CNN(nn.Module):
    def __init__(self, conv_layers:int=3, dropout_alpha:Union[int,list]=[0.25,0.25,0.25,0.5], batch_norm:bool=True):
        super(CIFAR10CNN, self).__init__()
        assert conv_layers + 1 == len(dropout_alpha) or isinstance(dropout_alpha, int), "dropout_alpha mismatch with the convultion layer numbers"
        self.conv_stack = nn.ModuleList()
        self.dropout_stack = nn.ModuleList()
        input_chan, output_chan = 3, 32
        # 第一层卷积，输入通道数 3，输出通道数 32，卷积核大小 3x3
        for i in range(conv_layers):
            dropout_l = nn.Dropout(dropout_alpha[i]) if isinstance(dropout_alpha, list) else nn.Dropout(dropout_alpha)
            self.dropout_stack.append(dropout_l)
            pair = nn.Sequential(
                    nn.Conv2d(in_channels=input_chan, out_channels=output_chan, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_chan)
                    )
            self.conv_stack.append(pair)
            # print(input_chan, output_chan)
            input_chan, output_chan = output_chan, output_chan * 2
        # 全连接层
        self.fc1 = nn.Linear(int(input_chan * 4 * 4 / 4**(conv_layers - 3)), 256)  # 输入大小是 128*4*4（经过多次池化缩小）
        self.bn_fc1 = nn.BatchNorm1d(256)
        if isinstance(dropout_alpha, int):
            self.dropout_fc1 = nn.Dropout(dropout_alpha)
        else:
            self.dropout_fc1 = nn.Dropout(dropout_alpha[-1])
        self.fc2 = nn.Linear(256, 10)  # CIFAR-10 有 10 个类别
        self.batch_norm = batch_norm

    def forward(self, x):
        for layer, dr in zip(self.conv_stack, self.dropout_stack):
            x = F.relu(layer(x))
            x = F.max_pool2d(x, 2)
            x = dr(x)

        # 将特征图展平为一维
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # 全连接层 1 + ReLU + Dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        # 全连接层 2（输出层）
        x = self.fc2(x)
        
        return x
from torchmetrics import Precision, Recall 
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
from tqdm import tqdm

def train(model_class, dataset: torch.utils.data.Dataset, test_loader: torch.utils.data.DataLoader, gpu_id: int = 0, optimizer:str = 'Adam', batch_size:int=64):
    lr_grid = [1e-2, 1e-3, 1e-4]
    epoch_grid = [9,12,15]
    conv_layer_grid = [3,4,5]
    # lr_grid = [1e-3,1e-4]
    # epoch_grid = [1,2,3]
    # conv_layer_grid = [3,4,5]
    kfold_log = {}
    for lr in lr_grid:
        for epoch_num in epoch_grid:
            for conv_layer in conv_layer_grid:
                print(lr, epoch_num, conv_layer)
                hyperparamter_key = str(lr) + '+' + str(epoch_num) + '+' + str(conv_layer)
                kfold_log[hyperparamter_key + ' validation performance'] = 0
                kfold_log[hyperparamter_key + ' test performance'] = 0
                # train_loader = torch.utils.data.DataLoader()
                K_F = KFold(n_splits=5, shuffle=True)
                
                for fold,(train_idx, val_idx) in enumerate(K_F.split(dataset)):
                    torch.cuda.empty_cache()
                    print(len(train_idx), len(val_idx))
                    train_subset = Subset(dataset, train_idx)
                    val_subset = Subset(dataset, val_idx)
                    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
                    model = CIFAR10CNN(conv_layers=conv_layer, dropout_alpha=[0.25] * (conv_layer) + [0.5])
                    model.train()
                    model = model.cuda(gpu_id)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
                    optimizer.zero_grad()
                    criterion = nn.CrossEntropyLoss()
                    for epoch in range(epoch_num):
                        with tqdm (train_loader, unit='batch') as t:
                            for X, y in t:
                                X, y = X.cuda(gpu_id), y.cuda(gpu_id)
                                optimizer.zero_grad()
                                y_pred = model(X)
                                loss = criterion(y_pred, y)
                                loss.backward()
                                optimizer.step()
                                t.set_postfix(loss=loss.item())
                            if (epoch + 1) % 3 == 0:
                                precision = Precision(task='multiclass',average='macro', num_classes=10).cuda(gpu_id)
                                recall = Recall(task='multiclass', average='macro', num_classes=10).cuda(gpu_id)
                                model.eval()
                                correct = 0
                                total = 0
                                for X,y in train_loader:
                                    X, y = X.cuda(gpu_id), y.cuda(gpu_id)
                                    y_pred = model(X)
                                    y_pred = torch.argmax(y_pred, dim=1)
                                    precision(y_pred, y)
                                    recall(y_pred, y)
                                    correct += (y_pred == y).sum().item()
                                    total += y.size(0)
                                _precision = precision.compute()
                                _recall = recall.compute()
                                f1 = 2 * _precision * _recall / (_precision + _recall)
                                print(f'Epoch {epoch + 1} Train Accuracy: {round(correct / total * 100, 2)}%\t\tPrecision: {round(_precision.item()*100, 2)}%\tRecall: {round(_recall.item()*100,2)}%\tF1: {round(f1.item()*100,2)}%')
                                precision = Precision(task='multiclass',average='macro', num_classes=10).cuda(gpu_id)
                                recall = Recall(task='multiclass', average='macro', num_classes=10).cuda(gpu_id)
                                correct = 0
                                total = 0
                                with torch.no_grad():
                                    for X, y in val_loader:
                                        X, y = X.cuda(gpu_id), y.cuda(gpu_id)
                                        y_pred = model(X)
                                        y_pred = torch.argmax(y_pred, dim=1)
                                        precision(y_pred, y)
                                        recall(y_pred, y)
                                        total += y.size(0)
                                        correct += (y_pred == y).sum().item()
                                    _precision = precision.compute()
                                    _recall = recall.compute()
                                    f1 = 2 * _precision * _recall / (_precision + _recall)
                                    print(f'Epoch {epoch + 1} Validation Accuracy: {round(correct / total * 100, 2)}%\tPrecision: {round(_precision.item()*100, 2)}%\tRecall: {round(_recall.item()*100,2)}%\tF1: {round(f1.item()*100,2)}%')
                                    if epoch == epoch_num - 1:
                                        kfold_log[hyperparamter_key + ' validation performance'] += round(f1.item()*100, 2)
                                # print(f'Epoch {epoch + 1} Validation Accuracy: {round(correct / total * 100, 2)}%')
                    with torch.no_grad():
                        model.eval()
                        precision = Precision(task='multiclass',average='macro', num_classes=10).cuda(gpu_id)
                        recall = Recall(task='multiclass', average='macro', num_classes=10).cuda(gpu_id)
                        total,correct = 0, 0
                        for X, y in test_loader:
                            X, y = X.cuda(gpu_id), y.cuda(gpu_id)
                            y_pred = model(X)
                            y_pred = torch.argmax(y_pred, dim=1)
                            precision(y_pred, y)
                            recall(y_pred, y)
                            total += y.size(0)
                            correct += (y_pred == y).sum().item()
                    _precision = precision.compute()
                    _recall = recall.compute()
                    f1 = 2 * _precision * _recall / (_precision + _recall)
                    print(f'Test Accuracy: {round(correct / total * 100, 2)}%\tPrecision: {round(_precision.item()*100, 2)}%\tRecall: {round(_recall.item()*100,2)}%\tF1: {round(f1.item()*100,2)}%')
                    kfold_log[hyperparamter_key + ' test performance'] += round(f1.item()*100, 2)
                kfold_log[hyperparamter_key + ' test performance'] /= 5
                kfold_log[hyperparamter_key + ' validation performance'] /= 5
    import json
    with open('./log.json','w') as f:
        json.dump(kfold_log, f)
    print(kfold_log)

if __name__ == '__main__':
    # 下载并加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(
        root='/raid_sdd/lwh/dataset/',  # 数据存储路径
        train=True,  # 训练集
        download=True,  # 如果数据集不存在，则下载
        transform=transform  # 预处理
    )
    testset = torchvision.datasets.CIFAR10(
        root='/raid_sdd/lwh/dataset/',
        train=False,  # 测试集
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    train(CIFAR10CNN, trainset, testloader, gpu_id=0, optimizer='Adam')