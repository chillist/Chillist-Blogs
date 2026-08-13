---
title: "DCGAN 的 MNIST 生成实践与训练结果分析"
slug: deep-learning-report-11
publishDate: 2026-07-24
description: "阅读 DCGAN 项目代码并完成 MNIST 训练实践，分析网络结构、训练循环和生成结果。"
---
## 一、学习概述

本周继续学习GAN相关内容，主要阅读了本项目中的DCGAN示例，并结合MNIST五轮训练产物，对数据读取、生成器、判别器、训练循环、模型保存和生成结果进行了整理。

上周的重点是阅读CycleGAN项目，认识两个图像域、两个生成器和两个判别器之间的调用关系。本周选择结构更紧凑的DCGAN，是希望重新回到一个生成器和一个判别器的基本对抗流程，把训练代码中的每一步看得更清楚。

和上周只进行代码阅读不同，本周项目目录中已经保存了MNIST数据集上的五轮训练结果。因此除了梳理`main.py`，我也对`real_samples.png`、`fake_samples_epoch_000.png`到`fake_samples_epoch_004.png`以及每轮保存的模型权重进行了观察。

目前我能够说明图像数据和随机噪声怎样进入网络、判别器和生成器怎样交替更新，也能从生成样本中看出训练过程存在明显波动。不过五轮训练仍然较短，项目没有单独保存损失日志，因此本周的结论主要来自代码阅读和生成图像的定性观察，还不能说明模型已经稳定收敛。

---

## 二、对DCGAN任务的初步认识

DCGAN的全称是Deep Convolutional Generative Adversarial Network，可以理解为使用卷积结构实现的生成对抗网络。项目README中说明，该示例实现了DCGAN论文中的基本思路，并与早期Torch版本的实现较为接近。

在本项目的MNIST任务中，生成器接收随机噪声，并输出一张64×64的单通道数字图像；判别器接收真实图像或生成图像，输出它属于真实数据的概率。

```
随机噪声 z
    ↓
生成器 netG
    ↓
生成数字 fake
    ↓
判别器 netD
    ↓
真假概率
```

DCGAN与基础GAN的目标没有改变，仍然是让生成器和判别器进行对抗。主要变化在于网络不再只使用全连接层，而是使用转置卷积逐步放大特征图，再使用普通卷积逐步压缩图像。这样更适合处理图像中的局部结构。

---

## 三、项目文件与运行参数整理

本周阅读的是项目中的Python版本`dcgan`。与上周的CycleGAN项目相比，它的结构更集中，数据集选择、模型定义和训练循环主要都写在`main.py`中。

```
dcgan
├── README.md
├── requirements.txt
├── main.py
├── data
│   └── MNIST
└── runs
    └── mnist_5epochs
        ├── real_samples.png
        ├── fake_samples_epoch_000.png
        ├── ...
        ├── fake_samples_epoch_004.png
        ├── netG_epoch_0.pth
        └── netD_epoch_0.pth
```

`main.py`使用`argparse`接收训练参数。结合代码中的默认值和`mnist_5epochs`目录，本次可以先按下面的配置理解。除训练轮数和输出目录外，其余项目若在运行时没有额外覆盖，就会采用代码中的默认值。

|参数|作用|本次整理值|
|---|---|---|
|dataset|选择数据集|mnist|
|imageSize|输入和输出图像尺寸|64|
|batchSize|每个批次的图像数量|64|
|nz|随机噪声向量维度|100|
|ngf / ndf|生成器、判别器基础通道数|64 / 64|
|niter|训练轮数|5|
|lr|Adam学习率|0.0002|
|beta1|Adam的第一个动量参数|0.5|
|outf|图像和权重输出目录|runs/mnist_5epochs|

对应的五轮训练命令可以整理为：

```
python main.py --dataset mnist --dataroot ./data \
  --niter 5 --outf ./runs/mnist_5epochs
```

---

## 四、MNIST数据读取与图像预处理

当`dataset`参数为`mnist`时，代码通过`torchvision.datasets.MNIST`自动下载和读取数据。MNIST原图为28×28的灰度图，进入网络前会被调整到64×64。

```
dataset = dset.MNIST(
    root=opt.dataroot,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

nc = 1
```

`Resize`负责统一图像尺寸，`ToTensor`将图像转换为PyTorch张量，`Normalize((0.5,), (0.5,))`将像素值从`[0,1]`转换到接近`[-1,1]`的范围。这样可以与生成器最后一层`Tanh`的输出范围保持一致。

MNIST是灰度图，因此代码将`nc`设置为1。这个变量会同时影响生成器最后一层的输出通道数和判别器第一层的输入通道数。若选择CIFAR-10、LSUN或文件夹图像，`nc`会设置为3。

```
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers)
)
```

训练时，`DataLoader`会打乱数据，并按批次返回图像和标签。DCGAN只使用`data[0]`中的图像，不使用MNIST的数字类别标签，因此该任务属于无监督生成。

---

## 五、运行设备、随机种子与权重初始化

程序启动后会设置随机种子。若没有手动传入`manualSeed`，代码会在1到10000之间随机选择一个值，再同时设置Python和PyTorch的随机状态。

```
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
```

设备选择部分默认使用CPU。只有传入`--accel`且PyTorch检测到可用加速器时，才会使用当前加速设备。

现有输出文件没有记录本次使用的设备和随机种子。后续若要比较不同实验，需要把这些信息一起保存。

生成器和判别器创建后都会调用`weights_init`进行初始化：

```
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
```

卷积层权重使用均值0、标准差0.02的正态分布。批归一化层权重以1为中心初始化，偏置初始化为0。

这部分让我认识到，生成模型不仅依赖网络结构，初始权重也可能影响前几轮对抗训练是否稳定。

---

## 六、生成器网络结构整理

生成器`Generator`接收形状为`[batch,100,1,1]`的随机噪声。网络连续使用五个`ConvTranspose2d`层，将1×1的潜在表示逐步放大为64×64图像。

```
[B, 100, 1, 1]
    ↓ ConvTranspose2d + BatchNorm + ReLU
[B, 512, 4, 4]
    ↓ ConvTranspose2d + BatchNorm + ReLU
[B, 256, 8, 8]
    ↓ ConvTranspose2d + BatchNorm + ReLU
[B, 128, 16, 16]
    ↓ ConvTranspose2d + BatchNorm + ReLU
[B, 64, 32, 32]
    ↓ ConvTranspose2d + Tanh
[B, 1, 64, 64]
```

前四个放大阶段使用`BatchNorm2d`和`ReLU`，最后一层不再使用批归一化，而是通过`Tanh`输出单通道图像。`Tanh`的输出范围与真实图像归一化后的范围对应。

根据第4轮`netG_epoch_4.pth`中的张量统计，生成器约有357万个可训练参数，权重文件大小约为14.3 MB。虽然MNIST图像较简单，网络仍然需要保存多层卷积核和批归一化参数。

---

## 七、判别器网络结构整理

判别器`Discriminator`的处理方向与生成器相反。它接收`[batch,1,64,64]`的图像，通过五个`Conv2d`层逐步减小空间尺寸，最后得到每张图像的真假概率。

```
[B, 1, 64, 64]
    ↓ Conv2d + LeakyReLU
[B, 64, 32, 32]
    ↓ Conv2d + BatchNorm + LeakyReLU
[B, 128, 16, 16]
    ↓ Conv2d + BatchNorm + LeakyReLU
[B, 256, 8, 8]
    ↓ Conv2d + BatchNorm + LeakyReLU
[B, 512, 4, 4]
    ↓ Conv2d + Sigmoid
[B]
```

判别器使用`LeakyReLU(0.2)`，使负数区域仍然保留一部分梯度。第一层没有使用批归一化，最后一层使用`Sigmoid`将结果限制在0到1之间。

`forward`最后通过下面的代码将输出整理为`[batch]`：

```
return output.view(-1, 1).squeeze(1)
```

这样可以与同样形状的真假标签计算二元交叉熵。

根据第4轮权重文件统计，判别器约有276万个可训练参数，权重文件大小约为11.1 MB。

---

## 八、损失函数与优化器

项目使用`BCELoss`作为生成器和判别器的损失函数，并用1表示真实、0表示生成。

```
criterion = nn.BCELoss()

real_label = 1
fake_label = 0
```

判别器训练时，真实图像的目标标签是1，生成图像的目标标签是0。生成器训练时会把生成图像的目标标签改为1，目的是让判别器把生成结果判断为真实。

两个网络分别使用Adam优化器，但学习率和动量参数相同：

```
optimizerD = optim.Adam(
    netD.parameters(),
    lr=opt.lr,
    betas=(opt.beta1, 0.999)
)

optimizerG = optim.Adam(
    netG.parameters(),
    lr=opt.lr,
    betas=(opt.beta1, 0.999)
)
```

默认学习率为0.0002，`beta1`为0.5。生成器和判别器必须使用不同的优化器，因为一次训练循环中需要先更新判别器，再单独更新生成器。

---

## 九、判别器更新过程

每个batch开始时先更新判别器。第一步让判别器识别真实MNIST图像，第二步让它识别生成器输出的假图像。

```
netD.zero_grad()

real_cpu = data[0].to(device)
batch_size = real_cpu.size(0)

label = torch.full(
    (batch_size,),
    real_label,
    dtype=real_cpu.dtype,
    device=device
)

output = netD(real_cpu)
errD_real = criterion(output, label)
errD_real.backward()

noise = torch.randn(
    batch_size,
    nz,
    1,
    1,
    device=device
)

fake = netG(noise)

label.fill_(fake_label)

output = netD(fake.detach())
errD_fake = criterion(output, label)
errD_fake.backward()

errD = errD_real + errD_fake
optimizerD.step()
```

这里的`fake.detach()`很重要。它会阻断生成图像到生成器的梯度，使判别器更新时只修改`netD`的参数，不会提前修改`netG`。

程序还记录了`D(x)`和第一次`D(G(z))`。

`D(x)`表示判别器对真实图像的平均判断，`D(G(z))`表示判别器对生成图像的平均判断。这些数值会打印到终端，但现有`runs`目录没有保存对应日志。

---

## 十、生成器更新过程

判别器更新完成后，程序再更新生成器。此时不会重新生成`fake`，而是继续使用当前batch中的生成图像。

```
netG.zero_grad()

label.fill_(real_label)

output = netD(fake)
errG = criterion(output, label)

errG.backward()
optimizerG.step()
```

这里没有使用`detach`，因此误差可以从判别器输出继续反向传播到生成器。

标签被改成真实标签1，表示生成器希望`netD(fake)`越接近1越好。

简化后的单批次训练顺序可以整理为：

```
真实图像
    ↓
更新判别器识别真实图像

随机噪声
    ↓
生成假图像
    ↓
假图像 detach
    ↓
更新判别器识别假图像

同一批假图像
    ↓
通过判别器计算损失
    ↓
更新生成器
```

这部分让我比上周更清楚地看到，所谓“交替训练”并不是同时修改两个网络，而是在一个batch内分成两个相互连接但梯度范围不同的步骤。

---

## 十一、训练输出与模型保存

程序在训练开始前创建`fixed_noise`。它只生成一次，之后始终使用同一批随机噪声，因此不同轮次的`fake_samples`图像可以按相同位置进行比较。

```
fixed_noise = torch.randn(
    opt.batchSize,
    nz,
    1,
    1,
    device=device
)
```

每训练100个iteration，程序会覆盖保存`real_samples.png`，并保存当前epoch对应的`fake_samples_epoch_XXX.png`。

```
if i % 100 == 0:
    vutils.save_image(
        real_cpu,
        "%s/real_samples.png" % opt.outf,
        normalize=True
    )

    fake = netG(fixed_noise)

    vutils.save_image(
        fake.detach(),
        "%s/fake_samples_epoch_%03d.png" % (
            opt.outf,
            epoch
        ),
        normalize=True
    )
```

由于同一轮内文件名不包含iteration，同一轮的图像会被多次覆盖，最终留下的是该轮最后一次触发保存时的结果。

每个epoch结束后，生成器和判别器都会保存一份`state_dict`：

```
torch.save(
    netG.state_dict(),
    "%s/netG_epoch_%d.pth" % (opt.outf, epoch)
)

torch.save(
    netD.state_dict(),
    "%s/netD_epoch_%d.pth" % (opt.outf, epoch)
)
```

当前目录中从`epoch_0`到`epoch_4`各有五份`netG`和`netD`权重，说明五轮训练的检查点都已经保留。

---

## 十二、MNIST前20轮训练结果观察

本次实验原计划训练50轮，输出目录设置为`runs/mnist_50epochs`。程序从0开始记录epoch，当前已经完整保存了`epoch_000`到`epoch_020`的生成图片和模型权重。训练进入`epoch_021`后只运行了9个batch，随后被手动中断。

本周对不同轮次使用固定噪声生成的图像进行了连续比较。由于`fixed_noise`在训练开始时只生成一次，因此不同epoch中相同位置的图片对应相同的潜在输入，可以直接观察生成器输出随训练过程发生的变化。

### 1. 第0轮到第13轮：数字结构基本稳定

从`epoch_000`到`epoch_013`，生成器输出的大部分图像都具有比较明显的MNIST数字特征。

第0轮已经能够生成黑色背景和白色笔画，但部分数字轮廓较模糊。随着训练继续，笔画逐渐变得清晰，背景中的灰色噪声也有所减少。

第8轮到第13轮的生成结果整体比较稳定。固定噪声在不同轮次中生成的数字类别和位置基本保持一致，说明生成器已经学习到了潜在向量与数字形状之间的一定对应关系。

不过，这几个轮次之间的提升并不明显，部分数字仍存在笔画断裂、粘连和难以辨认的问题。

  

### 2. 第14轮：生成结果突然退化为噪声

从`epoch_014`开始，生成图片出现了明显异常。原本可以辨认的数字结构几乎全部消失，64张生成图像都变成了灰白色的高频噪声。

  

第14轮的平均日志数据为：

```
Mean Loss_D: 0.28
Mean Loss_G: 4.78
Mean D(x): 0.92
Mean D(G(z)) before G update: 0.08
Mean D(G(z)) after G update: 0.05
```

该轮最后一个batch的输出为：

```
[14/50][937/938]
Loss_D: 0.0181
Loss_G: 7.6116
D(x): 0.9889
D(G(z)): 0.0045 / 0.0008
```

此时判别器几乎可以完全识别真实图像，同时把生成图像判断为假。生成器损失已经上升到7.6116，而生成器更新后`D(G(z))`仍然只有0.0008，说明生成器已经很难欺骗判别器。

值得注意的是，第14轮的平均损失还没有出现特别极端的数值，但生成图片已经完全失去数字结构。这说明在GAN训练中，图片质量可能比平均损失更早表现出训练异常。

### 3. 第15轮到第17轮：出现模式坍塌

第15轮以后，生成结果不再只是普通噪声，而是开始出现大量重复图案。

在`epoch_015`中，不同随机噪声几乎都生成了相同的白色图案；第16轮变成重复的亮点结构；第17轮则出现规则的横向条纹。

  

这种现象可以理解为模式坍塌，也就是生成器忽略了不同潜在向量之间的差异，把大量不同输入映射成几乎相同的输出。

这一阶段的日志变化也比较明显：

|程序轮次|平均Loss_D|平均Loss_G|平均D(x)|更新后的平均D(G(z))|
|---|---|---|---|---|
|epoch 15|约0.00|8.49|约1.00|约0.00|
|epoch 16|0.03|10.23|约1.00|约0.00|
|epoch 17|约0.00|36.45|约1.00|约0.00|

第17轮最后一个batch的结果为：

```
[17/50][937/938]
Loss_D: 0.0000
Loss_G: 36.8816
D(x): 1.0000
D(G(z)): 0.0000 / 0.0000
```

这个阶段判别器对真实图像的输出接近1，对生成图像的输出接近0。判别器已经占据明显优势，而生成器损失从8左右继续上升到36以上。

此时生成器虽然仍在更新，但已经无法从判别器中获得正常、有效的学习反馈，生成结果也逐渐从数字退化成重复纹理。

### 4. 第18轮到第20轮：损失和输出完全饱和

第18轮前半段仍然保持“判别器完全占优”的状态，但在第725个iteration附近，日志突然发生翻转。

```
[18/50][724/938]
Loss_D: 0.1916
Loss_G: 16.7615
D(x): 0.9841
D(G(z)): 0.1390 / 0.0000
```

紧接着变为：

```
[18/50][725/938]
Loss_D: 100.0022
Loss_G: 0.0000
D(x): 0.9979
D(G(z)): 1.0000 / 1.0000
```

从这一位置开始，判别器对真实图像和生成图像都输出接近1。由于假图像的目标标签是0，判别器对假图像的判断完全错误，因此`Loss_D`上升到100左右。

与此同时，生成器的目标标签是1，而判别器对生成图像的输出也正好是1，所以`Loss_G`下降到0。这并不表示生成器已经生成了高质量图片，只表示损失函数和判别器输出进入了完全饱和的异常状态。

  

第19轮和第20轮几乎始终保持下面的数值：

```
Loss_D: 100.0000
Loss_G: 0.0000
D(x): 1.0000
D(G(z)): 1.0000 / 1.0000
```

对应的生成图片也没有恢复数字结构，而是继续输出几乎完全相同的周期性亮点图案。

  

因此，不能把第18轮以后`Loss_G`等于0理解为生成器训练成功。结合生成图片可以确定，此时生成器和判别器已经失去正常的对抗关系，训练过程基本崩溃。

### 5. 前20轮结果总结

前20轮训练结果可以分为以下几个阶段：

|   |   |   |   |
|---|---|---|---|
|训练阶段|生成图片表现|日志特征|初步判断|
|epoch 0-13|能生成不同数字，轮廓逐渐清晰|损失存在波动，但仍能正常更新|相对稳定|
|epoch 14|数字突然变成高频噪声|判别器明显占优，生成器损失升高|生成结构崩溃|
|epoch 15-17|不同噪声生成相同或相似图案|D(x)接近1，D(G(z))接近0|模式坍塌|
|epoch 18后半段|重复亮点和周期纹理|Loss_D约100，Loss_G约0|损失发生饱和|
|epoch 19-20|图片不再恢复，结果基本不变|各项输出长期保持极端值|训练完全失效|

综合图片和日志来看，本次训练在第13轮以前能够生成较为正常的MNIST数字，第14轮开始发生明显退化，第15轮以后进入模式坍塌，第18轮后半段则进一步发展为判别器输出和损失函数的完全饱和。

因此，目前效果较好的权重应当从`netG_epoch_10.pth`到`netG_epoch_13.pth`之间选择，而不应该直接使用最后保存的模型。

---

## 十三、对第14轮以后结果异常的初步理解

第14轮以后生成结果全部变乱，并不是普通的“训练轮数增加但提升不明显”，而是一次比较明显的GAN训练不稳定现象。

首先，第14轮生成图片突然从数字变成噪声，说明生成器原来学习到的数字结构已经被破坏。

其次，第15轮以后，不同噪声输入几乎都生成相同图案。这符合模式坍塌的主要表现，即生成器失去输出多样性，只能生成少数甚至一种固定模式。

再次，从第15轮到第17轮，判别器对真实图像的输出接近1，对生成图像的输出接近0。判别器过于容易地区分真假图像，生成器损失不断增大，两个网络之间的训练强度已经明显失衡。

最后，第18轮第725个iteration附近，判别器对生成图像的输出突然从接近0跳到1。此后`Loss_D`约为100，`Loss_G`约为0，说明模型已经进入数值和输出同时饱和的异常状态。

目前还不能确定导致崩溃的唯一原因。可能的影响因素包括：

1. 判别器学习速度过快，导致生成器长期得不到有效梯度。
    
2. 使用`Sigmoid`和`BCELoss`时，判别器输出逐渐接近0或1，损失容易进入饱和区域。
    
3. 训练过程中没有根据生成质量进行早停或回退，模型崩溃后仍然继续更新。
    
4. 只运行了一组随机种子，暂时不能判断该现象是否能够稳定复现。
    
5. 保存固定噪声图片时，生成器仍处于训练模式，BatchNorm的运行统计量可能受到额外影响。
    

因此，报告中不能直接把异常归因于某一个参数，而应该先将其记录为“对抗训练失衡、模式坍塌和损失饱和”，再通过后续对照实验进一步确定原因。

---

## 十四、本周学习收获

本周除了继续阅读DCGAN代码，还完成了较长轮次的MNIST训练，并把生成图片与训练日志进行了对应分析。

通过前13轮结果，我观察到生成器可以逐渐学习MNIST数字的笔画、背景和整体形状。

通过第14轮以后的异常结果，我认识到GAN训练并不会随着epoch增加而持续改善。即使前面已经能够生成清晰数字，后续参数更新仍然可能破坏已经学习到的结构。

第15轮以后大量重复图案的出现，使我对模式坍塌有了更直观的理解。模式坍塌不仅表现为图片质量下降，更重要的是不同随机噪声失去了输出差异。

第18轮以后`Loss_D`约为100、`Loss_G`约为0，但生成图片仍然完全错误。这说明只查看损失大小可能得出相反结论，必须同时观察生成图片、`D(x)`和`D(G(z))`。

总体来看，本周最大的收获不是成功训练出了一个稳定的DCGAN，而是通过一次实际训练崩溃，认识了判别器占优、模式坍塌和损失饱和在日志和生成图片中的具体表现。

---

## 十五、后续学习安排

下一步首先不再继续使用第20轮权重，而是从第10轮到第13轮的检查点中选择生成效果相对正常的模型进行保存和比较。

之后准备固定`manualSeed`重新训练，并把每个iteration的`Loss_D`、`Loss_G`、`D(x)`和两个`D(G(z))`保存到CSV文件中，以便绘制完整曲线。

在训练参数方面，可以先尝试适当降低判别器学习率，或者分别设置生成器和判别器的学习率，使判别器不要过早占据绝对优势。

还可以尝试使用单边标签平滑，例如将真实标签从1调整为0.9，避免判别器过快输出完全确定的0或1。

损失函数方面，后续可以尝试使用`BCEWithLogitsLoss`，同时移除判别器最后的`Sigmoid`层，提高数值稳定性。

保存固定噪声生成结果时，计划临时将生成器切换到评估模式，并使用`torch.no_grad()`：

```
netG.eval()

with torch.no_grad():
    fixed_fake = netG(fixed_noise)

vutils.save_image(
    fixed_fake,
    image_path,
    normalize=True
)

netG.train()
```

后续重新训练时，还需要加入生成结果自动检查和早停机制。如果固定噪声生成图片从数字突然退化为噪声，就停止训练并保留前一轮正常模型，而不是继续更新到损失完全饱和。

通过这些对照实验，可以进一步判断本次崩溃主要来自判别器过强、损失函数饱和、随机初始化，还是其他训练参数问题。
