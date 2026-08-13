---
title: "从 AlexNet 到 DenseNet：CIFAR-10 分类性能对比"
slug: deep-learning-report-01
publishDate: 2026-05-13
description: "学习 AlexNet、VGG、ResNet 和 DenseNet，并在 CIFAR-10 上完成多模型训练与性能对比。"
---
## 一、学习概述

本阶段在前期掌握卷积神经网络基础结构、PyTorch 框架使用方法以及图像分类基本训练流程的基础上，继续学习了多种经典深度卷积神经网络，重点包括 AlexNet、VGG、ResNet 和 DenseNet。通过对不同网络结构的分析，进一步理解了卷积神经网络从普通卷积层堆叠，到更深层特征提取、残差连接与密集连接的发展思路。

理论学习方面，了解了 AlexNet 通过加深网络结构、引入 Dropout 等方式提升模型表达能力的设计思路，掌握了 VGG 网络中连续小卷积核堆叠的特点，理解了 ResNet 通过残差连接缓解深层网络训练困难的原理，并进一步学习了 DenseNet 中特征拼接、DenseBlock、TransitionLayer 等核心结构。

实践方面，本阶段基于 PyTorch 框架和 CIFAR-10 数据集，完成了 AlexNet、Mini-VGG、ResNet18 与 DenseNet 四种模型的搭建、训练和测试。实验中统一采用训练集、验证集和测试集划分方式，并将训练轮数设置为 **10 个 epoch**。最终，AlexNet 在测试集上取得了 **75.40%** 的准确率，Mini-VGG 取得了 **77.87%**，ResNet18 取得了 **85.18%**，DenseNet 取得了 **85.86%**。通过实验可以看出，随着网络结构不断优化，模型对图像特征的提取能力和分类性能也得到了明显提升。

---

## 二、具体学习内容
## （1）AlexNet 网络结构学习与模型实验

在进一步学习经典卷积神经网络时，首先了解了 AlexNet 的基本结构与设计思想。AlexNet 是较早在大规模图像分类任务中取得突出效果的深度卷积神经网络，相比早期浅层网络，它使用了更深的卷积结构，并引入 ReLU 激活函数、Dropout 等方法，使网络具备更强的特征提取能力和一定的抗过拟合能力。

AlexNet 的整体结构可以理解为：前半部分通过多层卷积层与池化层逐步提取图像特征，后半部分通过全连接层完成分类判断。其特点主要包括：

- 使用多层卷积网络逐步提取边缘、纹理和更高层语义特征；
- 使用 ReLU 激活函数增强非线性表达能力；
- 在分类器部分加入 Dropout，降低过拟合风险；
- 通过池化层压缩特征图尺寸，减少后续计算量。

由于 CIFAR-10 数据集图像尺寸为 `3×32×32`，与 AlexNet 原始使用的大尺寸图像不同，因此本实验对网络结构进行了适配。模型中采用 3×3 卷积核，并结合最大池化层逐步降低特征图尺寸，最终经过全连接层输出 10 类分类结果。

AlexNet-CIFAR10 的主要特征提取流程如下：

- 输入图像尺寸为 `3×32×32`；
- 经过第一组卷积和池化后，特征图尺寸变为 `64×16×16`；
- 经过第二组卷积和池化后，特征图尺寸变为 `192×8×8`；
- 后续继续使用多层卷积提取深层特征；
- 最后经过池化、展平和多层全连接层，输出 10 类分类结果。

![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-01.png)


**AlexNet 模型结构代码**
```python
self.features = nn.Sequential(  
    # 输入: [batch, 3, 32, 32]  
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
    nn.ReLU(inplace=True),  
    nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16  
  
    nn.Conv2d(64, 192, kernel_size=3, padding=1),  
    nn.ReLU(inplace=True),  
    nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8  
  
    nn.Conv2d(192, 384, kernel_size=3, padding=1),  
    nn.ReLU(inplace=True),  
  
    nn.Conv2d(384, 256, kernel_size=3, padding=1),  
    nn.ReLU(inplace=True),  
  
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  
    nn.ReLU(inplace=True),  
    nn.MaxPool2d(kernel_size=2, stride=2)   # 8 -> 4  
)
```

**AlexNet 分类器与前向传播代码**
```python
self.classifier = nn.Sequential(  
        nn.Dropout(0.5),  
        nn.Linear(256 * 4 * 4, 1024),  
        nn.ReLU(inplace=True),  
  
        nn.Dropout(0.5),  
        nn.Linear(1024, 512),  
        nn.ReLU(inplace=True),  
  
        nn.Linear(512, num_classes)  
    )  
  
def forward(self, x):  
    x = self.features(x)  
    x = torch.flatten(x, 1)  
    x = self.classifier(x)  
    return x
```


**AlexNet 训练结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-02.png)


**AlexNet 测试结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-03.png)

### （2）VGG 网络结构学习与 Mini-VGG 模型实验

VGG 网络的主要特点是使用多个连续的 3×3 小卷积层进行特征提取，并通过最大池化层逐步降低特征图尺寸。与早期网络相比，VGG 的结构更加规整，不同层之间的组织方式更加统一，便于分析网络中通道数和特征图大小的变化。

本阶段参考 VGG 的结构思想，搭建了适配 CIFAR-10 数据集的 Mini-VGG 网络。该模型输入为 3×32×32 的彩色图像，主体部分由 5 个卷积块构成。前两个卷积块各包含 2 个卷积层，后面三个卷积块各包含 3 个卷积层。每个卷积层后均加入 BatchNorm 和 ReLU 激活函数，每个卷积块末尾通过 MaxPool 进行下采样。

Mini-VGG 的特征提取过程大致如下：

- 输入图像尺寸为 3×32×32；
    
- 经过第一、二个卷积块后，特征图尺寸逐渐变为 16×16 和 8×8；
    
- 经过后续卷积块后，通道数逐步增加，特征图尺寸继续减小；
    
- 最终特征图被压缩为 256×1×1；
    
- 展平后输入全连接层，完成 10 类分类任务。
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-04.png)


**Mini-VGG 模型结构代码**
```python
# Mini-VGG 模型结构代码
self.features = nn.Sequential(  
    # Block 1: 32x32 -> 16x16  
    nn.Conv2d(3, 32, kernel_size=3, padding=1),  
    nn.BatchNorm2d(32),  
    nn.ReLU(),  
  
    nn.Conv2d(32, 32, kernel_size=3, padding=1),  
    nn.BatchNorm2d(32),  
    nn.ReLU(),  
  
    nn.MaxPool2d(kernel_size=2, stride=2),  
  
  
    # Block 2: 16x16 -> 8x8  
    nn.Conv2d(32, 64, kernel_size=3, padding=1),  
    nn.BatchNorm2d(64),  
    nn.ReLU(),  
  
    nn.Conv2d(64, 64, kernel_size=3, padding=1),  
    nn.BatchNorm2d(64),  
    nn.ReLU(),  
  
    nn.MaxPool2d(kernel_size=2, stride=2),  
  
  
    # Block 3: 8x8 -> 4x4  
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  
    nn.BatchNorm2d(128),  
    nn.ReLU(),  
  
    nn.Conv2d(128, 128, kernel_size=3, padding=1),  
    nn.BatchNorm2d(128),  
    nn.ReLU(),  
  
    nn.Conv2d(128, 128, kernel_size=3, padding=1),  
    nn.BatchNorm2d(128),  
    nn.ReLU(),  
  
    nn.MaxPool2d(kernel_size=2, stride=2),  
  
  
    # Block 4: 4x4 -> 2x2  
    nn.Conv2d(128, 256, kernel_size=3, padding=1),  
    nn.BatchNorm2d(256),  
    nn.ReLU(),  
  
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  
    nn.BatchNorm2d(256),  
    nn.ReLU(),  
  
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  
    nn.BatchNorm2d(256),  
    nn.ReLU(),  
  
    nn.MaxPool2d(kernel_size=2, stride=2),  
  
  
    # Block 5: 2x2 -> 1x1  
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  
    nn.BatchNorm2d(256),  
    nn.ReLU(),  
  
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  
    nn.BatchNorm2d(256),  
    nn.ReLU(),  
  
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  
    nn.BatchNorm2d(256),  
    nn.ReLU(),  
  
    nn.MaxPool2d(kernel_size=2, stride=2)  
)
```


**Mini-VGG 分类器与前向传播代码**
``` python
self.classifier = nn.Sequential(  
        nn.Flatten(),  
  
        # 经过 5 次池化后:  
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1        
        # 最终特征图大小: [batch_size, 256, 1, 1]  
        nn.Linear(256 * 1 * 1, 512),  
        nn.ReLU(),  
        nn.Dropout(0.5),  
  
        nn.Linear(512, 128),  
        nn.ReLU(),  
        nn.Dropout(0.3),  
  
        nn.Linear(128, 10)  
    )  
  
def forward(self, x):  
    x = self.features(x)  
    x = self.classifier(x)  
    return x
```


**Mini-VGG 训练结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-05.png)

**Mini-VGG 测试结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-06.png)

Mini-VGG 的实验结果说明，连续堆叠多个小卷积层能够增强网络对局部图像特征的提取能力。相比前期搭建的基础 CNN，Mini-VGG 结构更深、通道数变化更丰富，因此在 CIFAR-10 图像分类任务中取得了更好的测试效果。

---

### （3）ResNet18 残差网络学习与模型实验

随着卷积神经网络不断加深，普通网络可能会出现训练困难、梯度传播不充分以及网络退化等问题。ResNet 通过引入残差连接，使网络不再单纯学习完整映射，而是学习输入与目标输出之间的残差关系。其核心形式可以表示为：

[  
输出 = F(x) + x  
]

其中，(F(x)) 表示卷积分支学习到的特征，(x) 表示原始输入特征。通过将输入直接与卷积分支输出相加，可以在前向传播中保留原始信息，在反向传播中让梯度更容易向前传递。

本阶段搭建了适配 CIFAR-10 的 ResNet18 模型。由于 CIFAR-10 图片尺寸较小，模型没有直接采用原版 ResNet18 中 7×7 大卷积核和最前面的最大池化层，而是改为 3×3 卷积作为初始层，并去掉最前面的 MaxPool，从而避免特征图在网络前部被过快压缩。

ResNet18 主要由以下结构组成：

- 初始卷积层 b1；
    
- b2、b3、b4、b5 四个残差阶段；
    
- 每个阶段包含 2 个 Residual 残差块；
    
- 后续通过全局平均池化、展平和全连接层完成分类。
    

其中，Residual 残差块是 ResNet18 的核心模块。一个普通残差块包含两层卷积：

- 第一层：Conv → BatchNorm → ReLU；
    
- 第二层：Conv → BatchNorm；
    
- 最后将卷积分支输出与输入相加，再经过 ReLU 激活。
    

当输入与输出的通道数或空间尺寸不一致时，需要使用 1×1 卷积对捷径分支进行调整，使其能够与主分支结果相加。
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-07.png)


**Residual 残差块与resnet_block代码**
```python
class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels,  
                 use_1x1conv=False, strides=1):  
        super().__init__()  
  
        self.conv1 = nn.Conv2d(  
            input_channels,  
            num_channels,  
            kernel_size=3,  
            padding=1,  
            stride=strides  
        )  
  
        self.conv2 = nn.Conv2d(  
            num_channels,  
            num_channels,  
            kernel_size=3,  
            padding=1  
        )  
  
        if use_1x1conv:  
            self.conv3 = nn.Conv2d(  
                input_channels,  
                num_channels,  
                kernel_size=1,  
                stride=strides  
            )  
        else:  
            self.conv3 = None  
  
        self.bn1 = nn.BatchNorm2d(num_channels)  
        self.bn2 = nn.BatchNorm2d(num_channels)  
  
    def forward(self, X):  
        Y = F.relu(self.bn1(self.conv1(X)))  
        Y = self.bn2(self.conv2(Y))  
  
        if self.conv3:  
            X = self.conv3(X)  
  
        Y += X  
        return F.relu(Y)  
  
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):  
    blk = []  
  
    for i in range(num_residuals):  
        if i == 0 and not first_block:  
            blk.append(  
                Residual(input_channels, num_channels,  
                         use_1x1conv=True, strides=2)  
            )  
        else:  
            blk.append(  
                Residual(num_channels, num_channels)  
            )  
  
    return blk
```


**ResNet18 主体结构代码**
```python
class ResNet18(nn.Module):  
    def __init__(self, num_classes=10):  
        super().__init__()  
  
        # CIFAR-10 输入: [batch_size, 3, 32, 32]  
            nn.Conv2d(  
                in_channels=3,  
                out_channels=64,  
                kernel_size=3,  
                stride=1,  
                padding=1,  
                bias=False  
            ),  
            nn.BatchNorm2d(64),  
            nn.ReLU()  
        )  
  
        # 四个残差阶段  
        self.b2 = nn.Sequential(  
            *resnet_block(64, 64, 2, first_block=True)  
        )  
  
        self.b3 = nn.Sequential(  
            *resnet_block(64, 128, 2)  
        )  
  
        self.b4 = nn.Sequential(  
            *resnet_block(128, 256, 2)  
        )  
  
        self.b5 = nn.Sequential(  
            *resnet_block(256, 512, 2)  
        )  
  
        # 全局平均池化 + 分类层  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()  
        self.fc = nn.Linear(512, num_classes)  
  
    def forward(self, X):  
        X = self.b1(X)  
        X = self.b2(X)  
        X = self.b3(X)  
        X = self.b4(X)  
        X = self.b5(X)  
  
        X = self.avgpool(X)  
        X = self.flatten(X)  
        X = self.fc(X)  
  
        return X
```

ResNet18 训练完成后，最佳验证集准确率达到 89.78%，最终测试集准确率达到 89.33%。与 Mini-VGG 相比，ResNet18 的准确率有明显提升。


**ResNet18 训练结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-08.png)


**ResNet18 测试结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-09.png)


实验结果说明，残差连接能够有效改善深层卷积网络的训练效果。相比单纯依靠卷积层堆叠的 Mini-VGG，ResNet18 在更深的网络结构下仍然保持了较好的优化能力，因此在 CIFAR-10 分类任务中取得了更高的准确率。

---

### （4）DenseNet 密集连接网络学习与模型实验

在学习 ResNet 后，本阶段继续学习了 DenseNet 密集连接网络。DenseNet 与 ResNet 都希望改善深层网络中的信息传递问题，但二者采用的方式不同。ResNet 通过特征相加建立残差连接，而 DenseNet 则通过特征拼接实现密集连接。

DenseNet 的核心思想是：

- 每一层都接收前面所有层的输出作为输入；
    
- 当前层提取到的新特征，会继续与旧特征拼接后传递给后续层；
    
- 这种结构增强了特征复用能力，也让梯度传播路径更加丰富。
    

在代码实现中，DenseNet 使用：

```python
torch.cat([x, new_feature], dim=1)
```

在通道维度上对旧特征和新特征进行拼接。

DenseNet 的主要模块包括：

- **DenseLayer**：生成新的特征图，并与输入特征拼接；
    
- **DenseBlock**：由多个 DenseLayer 组成，通道数会随着层数增加逐渐增长；
    
- **TransitionLayer**：用于压缩通道数，并通过平均池化降低特征图尺寸。
    

本阶段实现的 DenseNet-CIFAR10 模型采用：

- `growth_rate = 16`
    
- `block_config = (6, 12, 24, 16)`
    
- `compression = 0.5`
    

模型整体结构为：

- 初始卷积层；
    
- DenseBlock1；
    
- Transition1；
    
- DenseBlock2；
    
- Transition2；
    
- DenseBlock3；
    
- Transition3；
    
- DenseBlock4；
    
- 全局平均池化；
    
- 全连接分类层。
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-10.png)


**DenseLayer 代码**
```python
class DenseLayer(nn.Module):  
    """  
    DenseNet 中的基本层：  
    BN -> ReLU -> 1x1 Conv    BN -> ReLU -> 3x3 Conv    
    最后将新特征与输入 x 在通道维度拼接  
    """  
    def __init__(  
        self,  
        in_channels,  
        growth_rate,  
        bn_size=4,  
        drop_rate=0.0  
    ):  
        super().__init__()  
  
        # 1x1 卷积：瓶颈层，先把通道数调整到 bn_size * growth_rate        
        self.bn1 = nn.BatchNorm2d(in_channels)  
        self.conv1 = nn.Conv2d(  
            in_channels,  
            bn_size * growth_rate,  
            kernel_size=1,  
            stride=1,  
            bias=False  
        )  
  
        # 3x3 卷积：真正生成 growth_rate 个新特征图  
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)  
        self.conv2 = nn.Conv2d(  
            bn_size * growth_rate,  
            growth_rate,  
            kernel_size=3,  
            stride=1,  
            padding=1,  
            bias=False  
        )  
  
        self.drop_rate = drop_rate  
  
    def forward(self, x):  
        # BN -> ReLU -> 1x1 Conv  
        out = self.conv1(F.relu(self.bn1(x), inplace=True))  
  
        # BN -> ReLU -> 3x3 Conv  
        out = self.conv2(F.relu(self.bn2(out), inplace=True))  
  
        # 可选 Dropout        if self.drop_rate > 0:  
            out = F.dropout(out, p=self.drop_rate, training=self.training)  
  
        # 在通道维度拼接  
        # x:   [B, in_channels, H, W]  
        # out: [B, growth_rate, H, W]        
        # 拼接后: [B, in_channels + growth_rate, H, W]  
        out = torch.cat([x, out], dim=1)  
  
        return out
```


**DenseBlock 与 TransitionLayer 代码**
```python
class DenseBlock(nn.Module):  
    """  
    一个 DenseBlock 由多个 DenseLayer 组成  
    每经过一层，通道数都会增加 growth_rate    
    """  
    def __init__(  
        self,  
        num_layers,  
        in_channels,  
        growth_rate,  
        bn_size=4,  
        drop_rate=0.0  
    ):  
        super().__init__()  
  
        layers = []  
        current_channels = in_channels  
  
        for _ in range(num_layers):  
            layer = DenseLayer(  
                in_channels=current_channels,  
                growth_rate=growth_rate,  
                bn_size=bn_size,  
                drop_rate=drop_rate  
            )  
            layers.append(layer)  
  
            # 每经过一个 DenseLayer，通道数增加 growth_rate            
            current_channels += growth_rate  
  
        self.block = nn.Sequential(*layers)  
  
    def forward(self, x):  
        return self.block(x)  
  
  
  
class TransitionLayer(nn.Module):  
    """  
    DenseBlock 之间的过渡层：  
    BN -> ReLU -> 1x1 Conv -> AvgPool2d  
    作用：  
    1. 压缩通道数  
    2. 特征图尺寸减半  
    """  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
  
        self.bn = nn.BatchNorm2d(in_channels)  
        self.conv = nn.Conv2d(  
            in_channels,  
            out_channels,  
            kernel_size=1,  
            stride=1,  
            bias=False  
        )  
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  
  
    def forward(self, x):  
        x = self.conv(F.relu(self.bn(x), inplace=True))  
        x = self.pool(x)  
        return x
```


**DenseNet 主体结构代码**
```python
class DenseNetCIFAR10(nn.Module):  
    """  
    适配 CIFAR-10 的 DenseNet-BC 风格网络  
  
    输入：  
        [B, 3, 32, 32]  
    输出：  
        [B, 10]    """  
    def __init__(  
        self,  
        growth_rate=16,  
        block_config=(6, 12, 24, 16),  
        num_init_features=32,  
        bn_size=4,  
        compression=0.5,  
        drop_rate=0.0,  
        num_classes=10  
    ):  
        super().__init__()  
  
        # -------------------------------------------------  
        # 初始卷积层  
        # CIFAR-10 图片较小，使用 3x3 卷积，不做初始池化  
        # 输入:  [B, 3, 32, 32]  
        # 输出:  [B, 32, 32, 32]  
        # -------------------------------------------------        
        self.stem = nn.Sequential(  
            nn.Conv2d(  
                3,  
                num_init_features,  
                kernel_size=3,  
                stride=1,  
                padding=1,  
                bias=False  
            ),  
            nn.BatchNorm2d(num_init_features),  
            nn.ReLU(inplace=True)  
        )  
  
        current_channels = num_init_features  
  
        # -------------------------------------------------  
        # DenseBlock 1        
        # -------------------------------------------------        
        self.block1 = DenseBlock(  
            num_layers=block_config[0],  
            in_channels=current_channels,  
            growth_rate=growth_rate,  
            bn_size=bn_size,  
            drop_rate=drop_rate  
        )  
        current_channels = current_channels + block_config[0] * growth_rate  
  
        transition1_channels = int(current_channels * compression)  
        self.transition1 = TransitionLayer(  
            current_channels,  
            transition1_channels  
        )  
        current_channels = transition1_channels  
  
        # -------------------------------------------------  
        # DenseBlock 2       
        # -------------------------------------------------        
         self.block2 = DenseBlock(  
            num_layers=block_config[1],  
            in_channels=current_channels,  
            growth_rate=growth_rate,  
            bn_size=bn_size,  
            drop_rate=drop_rate  
        )  
        current_channels = current_channels + block_config[1] * growth_rate  
  
        transition2_channels = int(current_channels * compression)  
        self.transition2 = TransitionLayer(  
            current_channels,  
            transition2_channels  
        )  
        current_channels = transition2_channels  
  
        # -------------------------------------------------  
        # DenseBlock 3        
        # -------------------------------------------------        
        self.block3 = DenseBlock(  
            num_layers=block_config[2],  
            in_channels=current_channels,  
            growth_rate=growth_rate,  
            bn_size=bn_size,  
            drop_rate=drop_rate  
        )  
        current_channels = current_channels + block_config[2] * growth_rate  
  
        transition3_channels = int(current_channels * compression)  
        self.transition3 = TransitionLayer(  
            current_channels,  
            transition3_channels  
        )  
        current_channels = transition3_channels  
  
        # -------------------------------------------------  
        # DenseBlock 4        
        # -------------------------------------------------        
        self.block4 = DenseBlock(  
            num_layers=block_config[3],  
            in_channels=current_channels,  
            growth_rate=growth_rate,  
            bn_size=bn_size,  
            drop_rate=drop_rate  
        )  
        current_channels = current_channels + block_config[3] * growth_rate  
  
        # 最后的 BN        
        self.bn_final = nn.BatchNorm2d(current_channels)  
  
        # 全局平均池化  
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
  
        # 分类层  
        self.fc = nn.Linear(current_channels, num_classes)  
  
    def forward(self, x):  
        # 输入: [B, 3, 32, 32]  
        x = self.stem(x)  
  
        x = self.block1(x)  
        x = self.transition1(x)  
  
        x = self.block2(x)  
        x = self.transition2(x)  
  
        x = self.block3(x)  
        x = self.transition3(x)  
  
        x = self.block4(x)  
  
        x = F.relu(self.bn_final(x), inplace=True)  
  
        x = self.global_avg_pool(x)  
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
  
        return x
```


**DenseNet 训练结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-11.png)


**DenseNet 测试结果**
![学习报告 1 配图](/assets/blog/deep-learning/report-01/image-12.png)


实验结果说明，DenseNet 通过密集连接和特征复用，能够有效提升模型对图像信息的利用能力。其验证集准确率略高于 ResNet18，测试集准确率与 ResNet18 基本接近，说明 DenseNet 在 CIFAR-10 分类任务中同样具有较强的特征表达能力。

---

### （4）CIFAR-10 数据集加载与预处理

本阶段实验统一使用 CIFAR-10 数据集进行模型训练和测试。该数据集包含 10 个类别，每张图像为 32×32 的彩色图片，输入形式为：

```text
3 × 32 × 32
```

在实验中，将原始训练数据划分为训练集和验证集，并使用官方测试集完成最终评估：

- 训练集：40000 张；
    
- 验证集：10000 张；
    
- 测试集：10000 张。
    

为提升模型的泛化能力，训练集使用了如下数据增强与预处理方式：

- `RandomCrop(32, padding=4)`：对图像进行随机裁剪，增加样本的空间变化；
    
- `RandomHorizontalFlip()`：随机水平翻转图像，增加训练样本多样性；
    
- `ToTensor()`：将图像转换为 PyTorch 张量；
    
- `Normalize()`：对图像数据进行标准化，使训练过程更加稳定。
    

验证集和测试集不进行随机增强，只保留张量转换与标准化处理，保证模型评估结果具有稳定性。


** CIFAR-10 数据加载与预处理代码**
```python
def get_cifar10_loaders(  
    data_dir="./data",  
    batch_size=64,  
    val_ratio=0.2,  
    seed=42  
):  
    """  
    返回 train_loader、val_loader、test_loader、classes  
    """  
    # 训练集使用数据增强  
    train_transform = transforms.Compose([  
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(  
            mean=(0.5, 0.5, 0.5),  
            std=(0.5, 0.5, 0.5)  
        )  
    ])  
  
    # 验证集和测试集不使用随机增强  
    test_transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.Normalize(  
            mean=(0.5, 0.5, 0.5),  
            std=(0.5, 0.5, 0.5)  
        )  
    ])  
  
    # 设置不同 transform    
    full_train_aug = datasets.CIFAR10(  
        root=data_dir,  
        train=True,  
        download=False,  
        transform=train_transform  
    )  
  
    full_train_plain = datasets.CIFAR10(  
        root=data_dir,  
        train=True,  
        download=False,  
        transform=test_transform  
    )  
  
    test_dataset = datasets.CIFAR10(  
        root=data_dir,  
        train=False,  
        download=False,  
        transform=test_transform  
    )  
  
    total_size = len(full_train_aug)  
    train_size = int((1 - val_ratio) * total_size)  
    val_size = total_size - train_size  
  
    generator = torch.Generator().manual_seed(seed)  
    indices = torch.randperm(total_size, generator=generator).tolist()  
  
    train_indices = indices[:train_size]  
    val_indices = indices[train_size:]  
  
    train_dataset = Subset(full_train_aug, train_indices)  
    val_dataset = Subset(full_train_plain, val_indices)  
  
    train_loader = DataLoader(  
        train_dataset,  
        batch_size=batch_size,  
        shuffle=True  
    )  
  
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=batch_size,  
        shuffle=False  
    )  
  
    test_loader = DataLoader(  
        test_dataset,  
        batch_size=batch_size,  
        shuffle=False  
    )  
  
    classes = full_train_aug.classes  
  
    print("训练集数量:", len(train_dataset))  
    print("验证集数量:", len(val_dataset))  
    print("测试集数量:", len(test_dataset))  
    print("类别:", classes)  
  
    return train_loader, val_loader, test_loader, classes
```

---

### （5）模型训练与测试

本阶段的 AlexNet、Mini-VGG、ResNet18 和 DenseNet 均使用相同的基本训练流程。训练过程遵循深度学习分类任务的标准步骤：

- **前向传播**：将一个批次的图片输入模型，得到预测结果；
    
- **损失计算**：使用交叉熵损失函数计算预测值与真实标签之间的误差；
    
- **反向传播**：通过自动求导机制计算各层参数梯度；
    
- **参数更新**：使用 Adam 优化器根据梯度更新模型参数；
    
- **验证集评估**：每轮训练结束后，在验证集上计算损失和准确率；
    
- **最佳模型保存**：当验证集准确率高于此前结果时，保存当前模型权重；
    
- **测试集评估**：训练结束后，加载最佳模型，在测试集上计算最终分类结果。
    

实验中主要训练参数如下：

- Epoch：10；
    
- Batch Size：64；
    
- Optimizer：Adam；
    
- Loss：CrossEntropyLoss；
    
- 运行设备：CUDA。
    


**模型训练主流程代码**
```python
def main():  
    # =========================  
    # 2. 准备设备  
    # =========================  
  
    device = get_device()  
  
    # =========================  
    # 3. 加载数据  
    # =========================  
  
    train_loader, val_loader, test_loader, classes = get_cifar10_loaders(  
        data_dir="./data",  
        batch_size=batch_size,  
        val_ratio=0.2,  
        seed=42  
    )  
  
    # 打印一批数据的 shape    
    print_batch_shape(train_loader)  
  
    # =========================  
    # 4. 创建模型  
    # =========================  
  
    model = DenseNetCIFAR10().to(device)  
  
    # =========================  
    # 5. 损失函数和优化器  
    # =========================  
  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
  
    # =========================  
    # 6. 正式训练  
    # =========================  
  
    best_val_acc = 0.0  
  
    for epoch in range(epochs):  
        train_loss, train_acc = train_one_epoch(  
            model,  
            train_loader,  
            criterion,  
            optimizer,  
            device  
        )  
  
        val_loss, val_acc = evaluate(  
            model,  
            val_loader,  
            criterion,  
            device  
        )  
  
        print(  
            f"Epoch [{epoch + 1}/{epochs}] "  
            f"Train Loss: {train_loss:.4f}, "  
            f"Train Acc: {train_acc:.4f}, "  
            f"Val Loss: {val_loss:.4f}, "  
            f"Val Acc: {val_acc:.4f}"  
        )  
  
        # 保存验证集效果最好的模型  
        if val_acc > best_val_acc:  
            best_val_acc = val_acc  
            torch.save(model.state_dict(), model_save_path)  
            print("保存当前最佳模型")  
  
    print("训练结束")  
    print("最佳验证集准确率:", best_val_acc)  
    print("模型已保存到:", model_save_path)
```


**模型测试流程代码**
```python
def main():  
    # =========================  
    # 1. 准备设备  
    # =========================  
  
    device = get_device()  
  
    # =========================  
    # 2. 加载数据  
    # =========================  
  
    train_loader, val_loader, test_loader, classes = get_cifar10_loaders(  
        data_dir="./data",  
        batch_size=batch_size,  
        val_ratio=0.2,  
        seed=42  
    )  
  
    # =========================  
    # 3. 创建模型并加载参数  
    # =========================  
  
    model = DenseNetCIFAR10().to(device)  
  
    model.load_state_dict(  
        torch.load(model_save_path, map_location=device)  
    )  
  
    # =========================  
    # 4. 测试模型  
    # =========================  
  
    criterion = nn.CrossEntropyLoss()  
  
    test_loss, test_acc = evaluate(  
        model,  
        test_loader,  
        criterion,  
        device  
    )  
  
    print("最终测试集 Loss:", test_loss)  
    print("最终测试集 Accuracy:", test_acc)
```

---

## （6）训练与测试结果分析

本阶段 AlexNet、Mini-VGG、ResNet18 和 DenseNet 均采用 **10 个 epoch** 进行训练，实验结果如下表所示。

**表1 四种网络模型实验结果对比**

|模型|Best Val Acc|Test Loss|Test Acc|
|---|--:|--:|--:|
|AlexNet|75.35%|0.7185|75.40%|
|Mini-VGG|77.95%|0.6731|77.87%|
|ResNet18|85.14%|0.4515|85.18%|
|DenseNet|86.64%|0.4137|85.86%|


综合来看，在相同训练轮数下，**DenseNet 的整体表现最好，ResNet18 次之，Mini-VGG 与 AlexNet 相对较低**。实验结果表明，残差连接和密集连接相比传统卷积堆叠结构，更有利于提升深层卷积神经网络在 CIFAR-10 图像分类任务中的分类效果与泛化能力。

---

## 三、后续学习安排

后续将在本阶段经典卷积神经网络学习和 CIFAR-10 图像分类实验的基础上，继续开展更加贴近实际应用场景的深度学习任务。下一步将重点学习真实图片分类项目的完整流程，包括自定义数据集加载、图像预处理、迁移学习和预训练模型微调等内容。
