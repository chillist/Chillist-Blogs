---
title: "GAN 训练代码解析：数据流、梯度传播与 detach"
slug: deep-learning-report-09
publishDate: 2026-07-10
description: "结合 PyTorch 代码复习基础 GAN 的数据流、交替训练、梯度传播、detach 与优化器机制。"
---
## 一、本周学习概述

本周没有继续深入学习CycleGAN和StarGAN，而是重新回到基础GAN，对生成器、判别器、损失函数、梯度传播和参数更新过程进行了复习。

上周的学习重点是了解GAN的基本概念，知道GAN由生成器和判别器组成，也初步接触了CycleGAN、PatchGAN和StarGAN等模型。本周则把重点放在基础GAN的训练代码上，尝试弄清楚一个batch中的数据如何经过生成器和判别器，以及两个网络为什么需要分阶段进行反向传播和参数更新。

本周主要通过重新观看课程内容、对照PyTorch代码、标注张量维度和整理训练流程的方式进行学习。在阅读训练代码时，我分别分析了判别器训练阶段和生成器训练阶段的梯度传播方向，并重点复习了`detach()`和两个优化器的作用。

本周主要学习了以下内容：

```text
1. 重新梳理基础GAN的整体训练过程
2. 分析生成器和判别器的输入、输出
3. 理解生成器与判别器的不同训练目标
4. 复习detach()对梯度传播的影响
5. 分析基础GAN中常见张量的维度变化
6. 初步了解基础GAN与DCGAN的结构区别
```

---

## 二、GAN整体结构与数据流动

GAN中包含生成器Generator和判别器Discriminator两个网络。

生成器接收随机噪声，并将噪声逐步转换为生成图像。判别器需要同时接收真实图像和生成图像，并对输入图像进行真假判断。

GAN的整体数据流动过程如图所示。

![学习报告 9 配图](/Chillist-Blogs/assets/blog/deep-learning/report-09/image-01.png)

生成器的输入通常是一组随机噪声向量，输出是生成的假图像。判别器的输入既可以是真实图像，也可以是生成器输出的假图像，最终输出一个真假判断结果。

GAN的基本过程可以表示为：

```text
随机噪声z
    ↓
生成器G
    ↓
生成图像
    ↓
判别器D
    ↓
真假判断结果
```

判别器还需要接收数据集中的真实图像：

```text
真实图像 ─────┐
              ↓
           判别器D → 真假判断
              ↑
生成图像 ─────┘
```

GAN并不是通过一个损失函数同时更新全部参数，而是将训练过程分成两个阶段：

```text
第一阶段：训练判别器D
第二阶段：训练生成器G
```

两个阶段在每个batch中交替执行，使生成器和判别器不断调整自身参数。

---

## 三、生成器和判别器的输入输出

### 1.生成器的输入输出

基础GAN中的生成器通常可以使用全连接神经网络实现。生成器接收随机噪声向量，并通过多层线性变换生成图像数据。

简化后的生成器结构如下：

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
```

以MNIST数据集为例，每张图像的大小为：

```text
1×28×28
```

将图像展开后，图像维度为：

```text
img_dim = 1×28×28 = 784
```

假设一个batch中有64个样本，随机噪声的维度为100，则生成器中的张量变化为：

```text
随机噪声：[64,100]
    ↓
第一层线性变换：[64,256]
    ↓
第二层线性变换：[64,512]
    ↓
输出层：[64,784]
```

对应代码为：

```python
z = torch.randn(64, 100).to(device)
fake_images = generator(z)

print("noise:", z.shape)
print("fake_images:", fake_images.shape)
```

输出结果应类似于：

```text
noise: torch.Size([64,100])
fake_images: torch.Size([64,784])
```

这表示生成器一次接收64个随机噪声向量，并生成64张展开后的MNIST图像。

### 2.判别器的输入输出

判别器的输入是一批真实图像或生成图像，输出是每张图像对应的真假判断结果。

简化后的判别器结构如下：

```python
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
```

如果判别器一次接收64张展开后的MNIST图像，则张量维度变化为：

```text
输入图像：[64,784]
    ↓
第一层线性变换：[64,512]
    ↓
第二层线性变换：[64,256]
    ↓
输出结果：[64,1]
```

其中，64表示一个batch中有64张图像，最后的1表示每张图像输出一个真假判断结果。

---

## 四、判别器与生成器的交替训练

一个batch中的GAN训练过程可以分为判别器训练和生成器训练两个阶段，具体过程如图所示。
![学习报告 9 配图](/Chillist-Blogs/assets/blog/deep-learning/report-09/image-02.png)
### 1.判别器训练阶段

判别器的目标是将真实图像判断为真，将生成器产生的假图像判断为假。

因此，判别器训练阶段使用的标签为：

```text
真实图像 → 标签1
生成图像 → 标签0
```

核心代码如下：

```python
real_outputs = discriminator(real_images)
d_loss_real = criterion(real_outputs, real_labels)

fake_images = generator(z)
fake_outputs = discriminator(fake_images.detach())
d_loss_fake = criterion(fake_outputs, fake_labels)

d_loss = d_loss_real + d_loss_fake

optimizer_D.zero_grad()
d_loss.backward()
optimizer_D.step()
```

判别器损失由真实图像损失和生成图像损失两部分组成：

```text
D_loss = D_loss_real + D_loss_fake
```

执行反向传播后，`optimizer_D`只负责更新判别器中的参数。

### 2.生成器训练阶段

生成器的目标是生成能够骗过判别器的图像。

虽然生成器输出的图像属于假图像，但在训练生成器时，希望判别器把这些图像判断为真实图像。因此，生成器损失计算时使用的目标标签是1。

核心代码如下：

```python
fake_images = generator(z)
outputs = discriminator(fake_images)

g_loss = criterion(outputs, real_labels)

optimizer_G.zero_grad()
g_loss.backward()
optimizer_G.step()
```

生成器训练阶段可以理解为：

```text
生成图像
    ↓
输入判别器
    ↓
希望判别器输出接近1
    ↓
根据损失更新生成器参数
```

判别器在这一阶段负责提供判断结果，真正需要更新的是生成器参数。

---

## 五、detach()的作用

本周复习过程中，`detach()`是我重点分析的一个代码点。

在判别器训练阶段，生成器只负责提供假图像，本阶段不需要根据判别器损失更新生成器参数。因此，需要在假图像输入判别器前使用：

```python
fake_images.detach()
```

完整代码为：

```python
fake_outputs = discriminator(fake_images.detach())
```

`detach()`会将`fake_images`从原来的计算图中分离，使梯度传播到假图像后停止，不再继续传回生成器。

可以简单表示为：

```text
判别器损失
    ↓
判别器参数
    ×
不继续传回生成器
```

如果训练判别器时不使用`detach()`，生成器可能会产生本阶段不需要的梯度。虽然没有执行`optimizer_G.step()`时生成器参数不会被直接修改，但这会增加额外计算和显存开销，也容易使两个训练阶段的梯度关系变得不清楚。

因此，我目前对`detach()`的理解是：

```text
训练判别器时：使用detach()
训练生成器时：不能使用detach()
```

训练生成器时需要让梯度经过判别器的计算结果继续传回生成器。如果此时使用`detach()`，生成器就无法根据损失调整参数。

---

## 六、两个优化器的作用

GAN中通常需要分别为生成器和判别器定义优化器：

```python
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
```

其中：

```text
optimizer_G：负责更新生成器参数
optimizer_D：负责更新判别器参数
```

在判别器训练阶段，执行：

```python
optimizer_D.zero_grad()
d_loss.backward()
optimizer_D.step()
```

在生成器训练阶段，执行：

```python
optimizer_G.zero_grad()
g_loss.backward()
optimizer_G.step()
```

两个优化器分别管理两个网络的参数，使生成器和判别器能够按照各自的目标进行更新。

通过对这部分代码的分析，我认识到GAN虽然由两个网络组成，但两个网络并不是同时按照同一个目标训练。判别器希望提高真假判断能力，生成器则希望生成更容易被判断为真实的图像。

---

## 七、基础GAN与DCGAN的初步对比

本周还初步了解了基础GAN与DCGAN之间的区别。

基础GAN通常使用全连接层处理展开后的图像，而DCGAN在生成器和判别器中加入卷积结构，更适合处理图像中的空间信息。

|对比内容|基础GAN|DCGAN|
|---|---|---|
|生成器结构|全连接层|转置卷积层|
|判别器结构|全连接层|卷积层|
|图像表示|通常展开为一维向量|保留通道、高度和宽度|
|空间特征处理能力|相对较弱|更适合学习图像空间特征|
|适用任务|简单图像生成|更复杂的图像生成|

基础GAN中的MNIST图像通常会被展开为：

```text
[batch_size,784]
```

DCGAN中的图像一般保留四维结构：

```text
[batch_size,channel,height,width]
```

例如，MNIST图像可以表示为：

```text
[64,1,28,28]
```

RGB图像可以表示为：

```text
[64,3,64,64]
```

DCGAN生成器通常使用转置卷积逐步增大特征图尺寸：

```text
随机噪声
    ↓
较小的特征图
    ↓
转置卷积上采样
    ↓
特征图尺寸逐渐增大
    ↓
生成完整图像
```

目前我只是对DCGAN的整体结构进行了初步了解，还没有深入学习每一层转置卷积的尺寸计算和具体参数设置。

---

## 八、本周学习中存在的问题

在进一步分析训练代码后，我发现自己对两个网络交替训练、梯度传播范围和损失变化之间的关系还不够熟悉。

目前仍需要继续理解的问题主要包括：

```text
1. 生成器损失为什么使用真实标签1
2. detach()对计算图和梯度传播的具体影响
3. 两个优化器分别在哪个阶段发挥作用
4. 训练生成器时为什么仍然需要经过判别器
5. 判别器和生成器能力不平衡时会出现什么问题
6. 基础GAN与DCGAN在图像维度处理上的区别
```

对于CycleGAN、PatchGAN和StarGAN，本周暂时没有继续深入学习。基础GAN的训练过程还没有完全熟悉，如果直接进入更加复杂的模型，容易只记住网络名称和基本功能，却不能真正理解代码中的数据流动和梯度更新。

因此，现阶段需要先缩小学习范围，把基础GAN的训练流程和代码运行过程弄清楚。

---

## 九、本周学习收获

通过本周的学习，我对基础GAN的一次完整训练过程有了比上周更具体的认识。

上周主要知道生成器负责生成图像，判别器负责判断真假。本周进一步认识到，GAN的核心不仅是两个网络各自的功能，还包括两个网络如何交替进行损失计算、反向传播和参数更新。

在代码方面，我对生成器和判别器的输入输出维度有了初步认识。以MNIST数据集为例，随机噪声可以表示为`[64,100]`，生成器输出可以表示为`[64,784]`，判别器输出为`[64,1]`。

我也进一步理解了`detach()`在判别器训练阶段的作用。它可以切断假图像和生成器之间的梯度连接，使本阶段只计算并更新判别器参数。训练生成器时则不能使用`detach()`，否则生成器无法根据判别器的结果调整参数。

对于两个优化器，我已经能够区分`optimizer_D`和`optimizer_G`分别负责更新哪个网络。通过将两个训练阶段分开分析，GAN的代码结构比之前更加清楚。

本周还初步了解了DCGAN与基础GAN的区别。基础GAN主要使用全连接层处理展开后的图像，DCGAN则使用卷积和转置卷积处理图像，更适合学习图像中的局部特征和空间结构。

不过，目前我对GAN的掌握仍然处在基础阶段，尤其是损失值变化、训练稳定性和生成结果之间的关系还不够清楚，需要通过实际运行项目继续理解。

---

## 十、后续学习安排

下一阶段暂时不继续学习CycleGAN和StarGAN，而是先运行一个简单的MNIST GAN项目，把基础训练流程与实际代码对应起来。

后续计划按照以下顺序进行：

```text
运行MNIST基础GAN
    ↓
打印关键张量的维度
    ↓
保存不同epoch的生成结果
    ↓
观察生成图像的变化
    ↓
记录生成器和判别器损失
    ↓
分析损失值与生成效果的关系
    ↓
学习DCGAN卷积结构
    ↓
再继续学习CycleGAN和StarGAN
```

在运行项目时，计划重点打印以下内容：

```python
print("noise:", z.shape)
print("fake_images:", fake_images.shape)
print("real_outputs:", real_outputs.shape)
print("fake_outputs:", fake_outputs.shape)
print("d_loss:", d_loss.item())
print("g_loss:", g_loss.item())
```

通过打印张量尺寸、观察损失值并保存生成图像，可以将比较抽象的GAN训练过程转化为更加直观的数据变化和图像变化。

后续学习的重点不是继续增加新的模型名称，而是先真正看懂并运行一次完整的GAN训练过程。在基础GAN训练流程较为熟悉后，再继续学习DCGAN、CycleGAN和StarGAN等模型。
