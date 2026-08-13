---
title: "GAN 入门：生成器、判别器与对抗训练"
slug: deep-learning-report-08
publishDate: 2026-07-02
description: "学习 GAN 的生成器、判别器、对抗训练与损失函数，并初步了解 CycleGAN、PatchGAN 和 StarGAN。"
---
## 一、学习概述

这周我开始学习GAN相关内容，目前已经看完课程前四章。和前面学习的Transformer、Mamba、RWKV不太一样，GAN的学习方向主要偏向图像生成和图像转换。前面学习的模型更多是在处理文本序列，关注的是token之间的关系；GAN更关注的是模型如何生成图像，以及生成出来的图像如何尽量接近真实数据。

这周的学习重点主要放在GAN的基本概念和基本原理上，包括生成器、判别器、对抗训练、损失函数、数据读取、基础网络结构和训练流程。对于后面课程中提到的CycleGAN、PatchGAN和StarGAN，我目前只是做了初步了解，知道它们大概解决什么问题，还没有深入学习完整的网络结构和损失函数。

通过这一周的学习，我对GAN有了一个比较基础的认识。GAN不是单独训练一个模型，而是同时训练两个网络：一个生成器，一个判别器。生成器负责生成假图像，判别器负责判断图像是真实的还是生成出来的。两个网络在训练过程中互相对抗，生成器想骗过判别器，判别器想识别出真假图像。这个过程是GAN和普通分类模型最大的不同。

---

## 二、GAN基本概念理解

GAN全称是Generative Adversarial Network，中文叫生成对抗网络。这里面有两个关键词，一个是“生成”，一个是“对抗”。

“生成”指的是模型能够生成新的数据。例如在图像任务中，模型可以从随机噪声生成一张图片。刚开始生成出来的图片可能很乱，但是随着训练进行，生成器会逐渐学到真实数据的一些特征。

“对抗”指的是生成器和判别器之间存在竞争关系。生成器希望生成更真实的假图像，让判别器分不出来；判别器希望更准确地区分真实图像和假图像。

GAN中最核心的两个部分是：

```text
Generator：生成器，负责生成假样本
Discriminator：判别器，负责判断样本真假
```

我现在对GAN的理解可以概括为：

```text
生成器：尽量生成更真实的图像
判别器：尽量判断图像是真是假
训练过程：生成器和判别器不断互相对抗
```

这个结构和以前学习的分类模型不同。分类模型一般是输入一张图像，然后输出类别；GAN里面有两个模型，它们的训练目标还不一样。判别器越强，生成器就需要生成更真实的图片；生成器越强，判别器就需要更仔细地区分真假。

---

## 三、生成器代码理解

在基础GAN中，生成器的输入通常是一个随机噪声向量，输出是一张生成图像。简单理解就是：

```text
随机噪声 → 生成器 → 假图像
```

课程中基础生成器的结构可以先理解成多层神经网络。输入是一段随机向量，经过多层线性变换和激活函数之后，最后输出图像向量。

生成器代码可以简化为：

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img
```

这里我主要理解了几个地方。`noise_dim`表示随机噪声的维度，`img_dim`表示图像展开后的维度。生成器输入的不是一张真实图片，而是一段随机噪声。模型训练的目标就是让这段随机噪声经过网络后，变成一张尽量真实的图像。

最后一层使用`Tanh`，是为了让生成图像的数值范围和真实图像预处理后的范围保持一致。如果真实图像被归一化到`[-1,1]`，那么生成器输出也应该尽量在这个范围内。

这部分我目前的理解是：生成器本质上是在学习一个映射关系，把随机噪声映射成图像。

---

## 四、判别器代码理解

判别器的作用是判断输入图像是真实图像还是生成图像。它的输入是一张图像，输出是一个概率值。输出越接近1，表示越像真实图像；输出越接近0，表示越像生成图像。

判别器可以简单理解为：

```text
图像 → 判别器 → 真假概率
```

判别器代码可以简化为：

```python
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
```

这里最后一层使用`Sigmoid`，是因为判别器最终输出的是真假概率，范围在0到1之间。

通过生成器和判别器这两段代码，我对GAN的基本结构有了更清楚的认识：

```text
生成器负责造图
判别器负责判断真假
两个网络一起构成GAN
```

目前我还没有深入到复杂生成器结构，比如DCGAN中的反卷积结构，只是先理解了最基础的全连接版本。

---

## 五、GAN损失函数和训练目标理解

GAN训练时，判别器和生成器的目标是不一样的。

判别器的目标是把真实图像判断为真，把生成图像判断为假。也就是说：

```text
真实图像 → 标签1
生成图像 → 标签0
```

生成器的目标是骗过判别器。它希望自己生成的假图像被判别器判断为真实图像，所以生成器训练时希望：

```text
生成图像 → 判别器输出接近1
```

这一点一开始比较容易混。生成图像本身当然是假图像，但是训练生成器时，我们希望判别器把它当成真图像，所以生成器损失中会把目标标签设置为1。

判别器损失可以简化理解为：

```python
criterion = nn.BCELoss()

real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)

real_outputs = discriminator(real_images)
d_loss_real = criterion(real_outputs, real_labels)

z = torch.randn(batch_size, noise_dim).to(device)
fake_images = generator(z)
fake_outputs = discriminator(fake_images.detach())
d_loss_fake = criterion(fake_outputs, fake_labels)

d_loss = d_loss_real + d_loss_fake
```

生成器损失可以简化理解为：

```python
z = torch.randn(batch_size, noise_dim).to(device)
fake_images = generator(z)

outputs = discriminator(fake_images)

g_loss = criterion(outputs, real_labels)
```

这里我重点理解了一个地方：生成器训练时用的是`real_labels`，因为生成器希望判别器把生成图像判断为真。

---

## 六、detach的作用理解

这周我觉得比较关键的一个代码点是`detach()`。

在训练判别器时，会把生成器生成的假图像输入给判别器。但是这一步只想更新判别器，不想更新生成器，所以需要写：

```python
fake_outputs = discriminator(fake_images.detach())
```

`detach()`的作用是切断梯度传播。这样判别器在反向传播时，梯度不会继续传回生成器。

我现在可以这样理解：

```text
训练判别器时：生成器只负责提供假图像，不更新生成器
训练生成器时：需要让判别器的判断结果反向影响生成器
```

所以训练判别器时要用`detach()`，训练生成器时不能用`detach()`。

这一点对理解GAN训练流程很重要。如果不区分这两个阶段，生成器和判别器的参数更新就容易混在一起。

---

## 七、GAN训练流程理解

GAN训练一般分为两个阶段：先训练判别器，再训练生成器。每个batch中都会重复这个过程。

完整训练流程可以简化为：

```python
for epoch in range(num_epochs):
    for real_images in dataloader:
        real_images = real_images.view(real_images.size(0), -1).to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 训练判别器
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(z)

        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach())

        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
```

这段代码中有两个优化器：

```text
optimizer_D：更新判别器参数
optimizer_G：更新生成器参数
```

这说明GAN不是一个损失函数直接更新全部参数，而是分阶段更新两个网络。判别器先学习区分真假，生成器再根据判别器的反馈改进生成结果。

目前我对GAN训练流程的理解是：

```text
先让判别器学会分辨真假
再让生成器学会骗过判别器
两个过程不断交替进行
```

---

## 八、数据读取和预处理理解

除了模型结构，课程中也涉及到数据读取和预处理。图像数据不能直接送进模型，需要先统一大小、转换成Tensor，并进行归一化。

数据读取代码可以简化为：

```python
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if name.endswith(".jpg") or name.endswith(".png")
        ]

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = self.transform(image)
        return image
```

这里我主要理解了几个预处理步骤：

```text
Resize：统一图片尺寸
ToTensor：把图片转换成张量
Normalize：把像素值归一化
```

虽然数据读取不是GAN最核心的理论部分，但它会直接影响训练。如果真实图像归一化范围和生成器输出范围不一致，模型训练就容易出问题。

---

## 九、CycleGAN相关内容初步了解

在学习基础GAN之后，课程中也简单接触到了CycleGAN。相比基础GAN从随机噪声生成图像，CycleGAN更偏向图像到图像的转换任务，比如把一类图像转换成另一类图像风格。

目前我对CycleGAN还没有进行深入学习，只是初步知道它主要用于两个图像域之间的转换。例如一组图片属于A域，另一组图片属于B域，模型希望学习A到B、B到A的转换关系。和普通GAN相比，CycleGAN不只是生成一张假图像，而是更关注图像风格或图像域之间的变化。

课程中提到的两个生成器、两个判别器、循环一致性损失等内容，我目前只是知道它们是CycleGAN里的重要组成部分，还没有完全掌握每一部分的具体作用。后续如果继续学习CycleGAN，需要重点补充它的训练流程、损失函数和源码实现。

目前我对CycleGAN的阶段性理解是：

```text
基础GAN：从随机噪声生成图像
CycleGAN：尝试在两个图像域之间进行转换
```

这部分现在还只是概念层面的了解，还不能说已经掌握。

---

## 十、PatchGAN判别器初步了解

课程中还简单提到了PatchGAN判别器。普通判别器一般判断整张图像是真是假，而PatchGAN更关注图像中的局部区域。

我目前对PatchGAN的理解还比较基础，只知道它常用于图像转换任务。因为图像转换不只是整张图像整体看起来合理，局部纹理也很重要。例如风格转换时，局部纹理是否真实会影响最终效果。

目前可以先这样理解：

```text
普通判别器：更像判断整张图像真假
PatchGAN：更像判断图像局部区域真假
```

PatchGAN的具体网络结构和输出维度我还没有深入分析，后续需要结合源码进一步理解它为什么适合CycleGAN这类图像转换任务。

---

## 十一、StarGAN相关内容初步了解

后面课程中也简单接触到了StarGAN。相比CycleGAN主要处理两个图像域之间的转换，StarGAN更偏向多属性或多领域图像转换。比如在人脸图像中，可以通过目标属性标签控制生成结果，让模型把原图转换成指定属性的图像。

目前我对StarGAN也只是做了初步了解。简单来说，StarGAN的输入不只是原始图像，还会加入目标属性标签，让生成器知道应该往哪个方向转换。

可以简单理解为：

```text
原图 + 目标属性标签 → 生成器 → 转换后的图像
```

这一部分我还没有仔细学习完整源码，只是知道StarGAN和基础GAN的区别在于它加入了条件信息，也就是目标属性标签。它不是单纯从随机噪声生成图像，而是在原图基础上进行属性转换。

后续如果继续学习StarGAN，需要重点理解目标标签是如何处理的、标签如何和图像一起输入生成器，以及判别器为什么既要判断真假，又要判断属性类别。

---

## 十二、GAN、CycleGAN和StarGAN的阶段性认识

通过这周学习，我对GAN系列模型有了一个初步的整体认识。基础GAN主要解决的是从随机噪声生成图像的问题，核心是生成器和判别器之间的对抗训练。

CycleGAN和StarGAN可以看作是在GAN思想上的进一步扩展。CycleGAN更偏向两个图像域之间的转换，StarGAN更偏向多个属性之间的转换。不过目前我对它们还只是停留在概念层面的认识，没有深入掌握完整网络结构、损失函数和源码细节。

目前可以先这样理解：

```text
GAN：学习如何生成图像
CycleGAN：学习两个图像域之间的转换
StarGAN：学习根据目标属性进行图像转换
```

这周的重点还是GAN的基本概念和原理，CycleGAN和StarGAN只是先做了初步了解，后续还需要继续深入学习。

---

## 十三、本周学习收获

通过本周学习，我初步了解了GAN生成对抗网络的基本思想。GAN和之前学习的分类模型不太一样，它不是单独训练一个模型完成分类任务，而是同时训练生成器和判别器两个网络。生成器希望生成更接近真实数据的样本，判别器希望准确区分真实样本和生成样本，二者在对抗过程中不断优化。

在代码理解方面，我重点学习了基础GAN中生成器、判别器和训练流程。生成器的输入是随机噪声，输出是生成图像；判别器的输入是图像，输出是真假判断结果。训练时需要分别更新判别器和生成器，其中`detach()`的使用比较关键，它可以在训练判别器时阻断生成器的梯度更新，避免两个网络的参数更新混在一起。

对于CycleGAN和StarGAN，我目前主要是初步了解它们的任务目标和大致思路。CycleGAN用于图像域之间的转换，StarGAN用于多属性图像转换。它们都属于GAN思想的扩展应用，但具体网络结构和源码实现还需要后续继续学习。

总体来看，这周主要完成了GAN基础概念的入门学习，对生成器、判别器、对抗训练、损失函数和基本训练流程有了初步认识，也知道了CycleGAN和StarGAN是后续可以继续深入学习的两个方向。

---

## 十四、后续学习安排

后续将继续围绕GAN基础内容进行学习，重点复习生成器和判别器的训练流程，进一步理解判别器损失、生成器损失、梯度更新和`detach()`的作用。接下来还需要结合代码运行一个简单的GAN项目，观察生成图像在训练过程中的变化，加深对生成对抗训练的理解。

在掌握基础GAN后，再继续深入学习CycleGAN和StarGAN。对于CycleGAN，后续重点学习两个图像域之间是如何转换的，以及循环一致性损失为什么能够约束生成结果。对于StarGAN，后续重点学习目标属性标签如何输入模型，以及一个生成器如何完成多个属性之间的转换。

后续学习可以先从简单GAN或DCGAN开始，等基础训练流程比较熟悉后，再继续学习Pix2Pix、CycleGAN和StarGAN等更复杂的图像转换模型。
