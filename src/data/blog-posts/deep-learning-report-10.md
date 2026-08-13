---
title: "GAN 训练逻辑复盘：真假标签、损失与参数更新"
slug: deep-learning-report-10
publishDate: 2026-07-17
description: "系统复习 GAN 的基础结构和训练逻辑，整理真假标签、损失函数、detach 与参数更新流程。"
---
## 一、学习概述

本周没有继续观看GAN课程后面的新章节，学习内容主要以复习和整理上周接触到的基础知识为主。上周刚开始学习GAN时，一次接触了生成器、判别器、损失函数、数据读取和训练流程等多个概念，当时更多是跟着课程了解整体过程，对部分代码的作用还没有完全理顺。因此，这周没有急着继续学习CycleGAN和StarGAN，而是重新回顾了基础GAN的结构和训练逻辑。

这周主要重新整理了生成器和判别器之间的关系、两个网络各自的训练目标、真假标签的设置方式、两个优化器的作用，以及`detach()`在训练过程中的意义。同时也重新看了一遍简单GAN代码，对一轮训练中数据是如何流动的有了更清楚的认识。

虽然本周没有增加很多新的课程内容，但通过复习，我发现之前对GAN的理解有些地方比较模糊。例如，生成器生成的明明是假图像，为什么训练生成器时目标标签却设置为1；训练判别器时为什么需要使用`detach()`；为什么生成器和判别器不能使用同一个优化器直接一起更新。这些问题在重新梳理后比上周清楚了一些。

---

## 二、GAN整体结构复习

GAN全称为Generative Adversarial Network，中文叫生成对抗网络。它和之前学习的图像分类模型有较大区别。

普通图像分类模型一般是输入一张图片，然后输出这张图片所属的类别。模型的训练目标比较单一，就是让预测结果尽量接近真实标签。

GAN中同时包含两个网络：

```text
Generator：生成器
Discriminator：判别器
```

生成器负责生成假样本，判别器负责判断输入样本是真实的还是生成出来的。两个网络的目标不同，但又互相影响。

生成器的目标是：

```text
让自己生成的图像越来越接近真实图像，
使判别器无法准确判断图像是生成出来的。
```

判别器的目标是：

```text
尽量准确地区分真实图像和生成器生成的图像。
```

GAN的整体过程可以理解为：

```text
随机噪声
↓
生成器
↓
生成图像
↓
判别器
↓
输出真假判断结果
```

生成器和判别器会在训练过程中不断对抗。判别器判断得越准确，生成器就需要进一步提高生成质量；生成器生成得越真实，判别器也需要提高自己的判断能力。

通过重新整理，我觉得GAN中的“对抗”并不是两个网络互相独立地训练，而是一个网络的输出会成为另一个网络训练的一部分。生成器需要根据判别器给出的反馈改进，判别器也需要不断面对生成器产生的新假样本。

---

## 三、生成器作用复习

生成器的输入一般不是一张真实图像，而是一组随机噪声。随机噪声可以理解为一组随机生成的数字，模型需要将这些数字逐步转换成具有图像特征的数据。

简单生成器可以写成：

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
        return self.model(z)
```

这段代码中，`noise_dim`表示随机噪声的维度，`img_dim`表示生成图像展开后的维度。

生成器的输入输出关系可以写成：

```text
随机噪声z
↓
多层线性变换
↓
生成图像
```

这里的随机噪声本身没有图像意义。生成器通过训练，逐渐学习如何把不同噪声映射成不同的图像。

最后一层使用`Tanh`，主要是为了让生成器输出范围和真实图像预处理后的范围保持一致。假设真实图像已经归一化到`[-1,1]`，那么生成器输出也应该处于类似范围。

本周重新复习后，我对生成器的理解是：生成器不是在训练集中查找一张图像，也不是直接复制真实图像，而是在学习真实数据的大致分布，并根据随机噪声生成新的样本。

---

## 四、判别器作用复习

判别器的任务比较接近二分类。它接收一张图像，然后判断图像是真实图像还是生成器产生的假图像。

简单判别器可以写成：

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
        return self.model(img)
```

判别器的输入输出关系可以理解为：

```text
输入图像
↓
判别器
↓
输出真假概率
```

最后一层使用`Sigmoid`，把输出限制在0到1之间。输出越接近1，表示判别器越认为这是一张真实图像；输出越接近0，表示越认为它是生成图像。

本周在复习时，我进一步意识到判别器本身并不知道一张图像是由谁生成的，它只能根据训练过程中学到的图像特征进行判断。真实图像的特征越稳定，生成图像和真实图像之间差距越大，判别器就越容易区分。随着生成器能力提高，判别器的判断难度也会增加。

---

## 五、真假标签和损失函数整理

GAN的损失函数主要围绕真假判断进行，因此基础GAN中经常使用二分类交叉熵损失：

```python
criterion = nn.BCELoss()
```

训练判别器时，需要准备两类标签：

```python
real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)
```

其中：

```text
真实图像对应标签1
生成图像对应标签0
```

判别器对真实图像的损失为：

```python
real_outputs = discriminator(real_images)
d_loss_real = criterion(real_outputs, real_labels)
```

判别器对生成图像的损失为：

```python
fake_outputs = discriminator(fake_images.detach())
d_loss_fake = criterion(fake_outputs, fake_labels)
```

判别器总损失可以写成：

```python
d_loss = d_loss_real + d_loss_fake
```

这部分的目标比较直接：真实图像尽量判断为1，生成图像尽量判断为0。

生成器的训练目标容易和判别器混淆。生成器产生的图像虽然是假图像，但生成器希望它骗过判别器。因此训练生成器时，目标标签要使用1：

```python
outputs = discriminator(fake_images)
g_loss = criterion(outputs, real_labels)
```

这段代码表示生成器希望判别器把假图像判断成真实图像。

本周复习后，我把这个地方整理成：

```text
训练判别器时：
真实图像对应1，生成图像对应0。

训练生成器时：
希望生成图像被判别成1。
```

这样理解之后，生成器为什么使用`real_labels`就没有之前那么绕了。

---

## 六、detach作用重新理解

`detach()`是这周重点重新看的一个代码点。

训练判别器时，生成器会先生成假图像：

```python
z = torch.randn(batch_size, noise_dim).to(device)
fake_images = generator(z)
```

然后把假图像输入判别器：

```python
fake_outputs = discriminator(fake_images.detach())
```

这里使用`detach()`，是因为当前步骤只想训练判别器，不希望更新生成器参数。

如果不使用`detach()`，在执行反向传播时，梯度可能会沿着假图像继续传回生成器。这样原本只训练判别器的一步，就可能同时影响生成器。

可以把它理解为：

```text
fake_images原本连接着生成器的计算过程。

detach()之后，
fake_images只作为普通数据输入判别器，
不再把当前梯度传回生成器。
```

训练生成器时则不能使用`detach()`：

```python
fake_images = generator(z)
outputs = discriminator(fake_images)
g_loss = criterion(outputs, real_labels)
```

因为生成器需要根据判别器给出的判断结果进行参数更新。如果这时使用`detach()`，梯度就无法传回生成器，生成器也就无法学习。

这周重新整理后，可以将它概括为：

```text
训练判别器：使用detach，只更新判别器。
训练生成器：不使用detach，让梯度传回生成器。
```

---

## 七、两个优化器的作用整理

GAN中通常需要分别为生成器和判别器定义优化器：

```python
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
```

其中：

```text
optimizer_G只负责更新生成器参数
optimizer_D只负责更新判别器参数
```

训练判别器时：

```python
optimizer_D.zero_grad()
d_loss.backward()
optimizer_D.step()
```

训练生成器时：

```python
optimizer_G.zero_grad()
g_loss.backward()
optimizer_G.step()
```

`zero_grad()`用于清空上一轮保留下来的梯度，`backward()`用于计算当前损失对应的梯度，`step()`用于根据梯度更新模型参数。

以前学习普通分类模型时，一般只有一个模型和一个优化器。GAN中有两个网络，因此需要分别控制它们的参数更新。

通过本周复习，我对两个优化器的理解更清楚了。虽然生成器和判别器属于同一个GAN系统，但它们本身是两个独立网络，有各自的参数和训练目标，所以不能简单使用同一个优化器同时更新。

---

## 八、一轮GAN训练过程整理

一轮基础GAN训练可以分成两个阶段。

第一阶段是训练判别器。模型先读取真实图像，再使用生成器产生假图像。判别器分别判断真实图像和假图像，并根据两个结果计算损失。

第二阶段是训练生成器。生成器重新生成一批假图像，将其输入判别器，并根据判别结果计算生成器损失。

简化代码如下：

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

这一段代码可以概括成：

```text
真实图像和假图像训练判别器
↓
生成新的假图像
↓
利用判别器反馈训练生成器
↓
不断重复
```

这周没有实际运行完整项目，但通过逐步整理代码，我对一轮训练中生成器和判别器的执行顺序有了更清楚的认识。

---

## 九、数据预处理内容复习

本周也简单复习了图像数据进入GAN前的处理过程。真实图像需要经过统一尺寸、转换为Tensor和归一化等操作。

```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

其中：

```text
Resize：统一图片大小
ToTensor：把图像转换成PyTorch张量
Normalize：调整图像数值范围
```

如果生成器最后使用`Tanh`输出`[-1,1]`范围的图像，那么真实图像也应该归一化到相近范围。这样判别器接收到的真实图像和生成图像在数值范围上比较一致。

以前学习图像分类时也使用过数据预处理，但在GAN中，生成器输出范围和真实图像归一化范围之间的对应关系更加重要。

---

## 十、对GAN训练困难的初步认识

复习GAN训练流程时，我也注意到GAN训练和普通分类模型相比可能更不稳定。

普通分类模型一般只需要让一个损失逐渐下降，而GAN中同时存在生成器损失和判别器损失。两个网络的能力需要保持一定平衡。

如果判别器太强，生成器产生的图像很容易被识别，生成器可能难以获得有效反馈。如果生成器提升过快，判别器又可能无法准确区分真假图像。

目前我对这部分还没有结合实验深入分析，只是初步知道GAN训练过程中不能只看某一个损失是否下降，也不能简单认为生成器损失越低越好。后续需要结合生成图像的实际效果和两个网络的损失变化一起判断训练情况。

另外，课程中可能还会涉及模式崩溃等问题，但目前还没有深入学习。现在只知道GAN训练比普通分类网络更容易出现不稳定情况，需要在后续实验中继续观察。

---

## 十一、CycleGAN和StarGAN内容回顾

本周没有继续学习CycleGAN和StarGAN的新内容，只是简单回顾了它们和基础GAN之间的关系。

目前对CycleGAN的认识仍然停留在概念层面。它主要用于不同图像域之间的转换，例如把一种风格的图像转换成另一种风格。课程中提到了两个生成器、两个判别器和循环一致性等概念，但这些内容还没有深入掌握。

目前可以先理解为：

```text
基础GAN：从随机噪声生成图像
CycleGAN：尝试在两个图像域之间进行转换
```

对StarGAN的认识也比较基础。它主要和多属性图像转换有关，可以根据不同目标标签控制图像转换方向。但目标标签如何加入网络、生成器和判别器具体如何训练，目前还没有仔细学习。

本周没有继续展开这些内容，主要原因是基础GAN的训练过程还需要进一步消化。先把生成器、判别器和梯度更新理顺，再学习后面的模型会更容易一些。

---

## 十二、本周学习收获

本周虽然没有继续观看新章节，但通过重新整理基础GAN，我对生成器和判别器的关系比上周更加清楚。

生成器从随机噪声出发生成假图像，判别器接收真实图像和生成图像并判断真假。训练过程中，判别器先学习区分真假，生成器再根据判别器的反馈提高生成质量。

这周重点重新理解了真假标签的设置。训练判别器时，真实图像使用标签1，生成图像使用标签0；训练生成器时，虽然输入判别器的是假图像，但目标标签使用1，因为生成器希望骗过判别器。

`detach()`也是本周复习中比较重要的内容。训练判别器时使用`detach()`，可以阻止梯度传回生成器；训练生成器时不使用`detach()`，让判别结果能够影响生成器参数。

另外，我也进一步分清了两个优化器的作用。生成器和判别器拥有不同参数，需要分别计算损失、清空梯度和执行参数更新。

这周没有学习太多新知识，主要是把上周接触的内容重新消化。虽然学习进度不快，但对基础训练过程的理解比之前更加连贯。

---

## 十三、目前存在的问题

目前对GAN的整体结构已经有了初步认识，但还存在一些没有完全弄清楚的问题。

第一，对生成器损失和判别器损失的变化规律还不熟悉。目前只是知道两个网络需要交替训练，但还不会根据损失值判断训练状态是否正常。

第二，还没有实际运行一个完整的GAN项目。虽然能够大致看懂训练代码，但对训练日志、模型保存、生成图像保存和结果变化还缺少实际体验。

第三，对卷积生成器和卷积判别器还没有认真学习。目前整理的网络主要是全连接层结构，生成复杂图像时通常需要使用卷积网络，这部分需要后续补充。

第四，对CycleGAN和StarGAN仍然只是初步了解，还不能解释它们的完整结构和损失函数。

---

## 十四、后续学习安排

下一步计划继续复习基础GAN，并尝试运行一个较简单的项目。可以先使用MNIST等规模较小的数据集，观察模型从随机噪声逐渐生成数字图像的过程。

运行项目时，重点记录生成器损失、判别器损失和不同训练轮次的生成结果，尝试理解两个网络之间的平衡关系。

在基础GAN训练流程更加熟悉后，再学习DCGAN。DCGAN会使用卷积网络搭建生成器和判别器，比全连接GAN更适合图像生成任务。

CycleGAN和StarGAN暂时先不深入，等生成器、判别器、梯度传播和基础训练流程掌握得更牢固后，再继续学习图像转换相关模型。
