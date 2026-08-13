---
title: "CycleGAN 源码解析：双向图像转换与训练流程"
slug: deep-learning-report-13
publishDate: 2026-08-07
description: "阅读 CycleGAN 项目源码，梳理双向图像转换中的数据流、生成器、判别器和训练流程。"
---
## 一、学习概述

本周继续进行GAN相关内容的学习。在上周进一步分析DCGAN生成器、判别器训练过程和梯度更新机制的基础上，本周开始把学习重点转向更加完整的CycleGAN项目，主要阅读 `junyanz/pytorch-CycleGAN-and-pix2pix` 项目的源码，并结合之前学习的GAN基础理解CycleGAN中各个模块之间的关系。

与之前学习的DCGAN相比，CycleGAN的项目结构明显更加复杂。DCGAN主要围绕一个生成器和一个判别器展开，而CycleGAN中存在两个生成器和两个判别器，同时还包含数据集读取、参数配置、模型基类、训练与测试流程、模型保存以及生成结果可视化等模块。

本周没有急于完整训练CycleGAN，而是先从项目整体结构和关键代码入手，重点理解以下几个问题：

1. CycleGAN项目中各个文件分别负责什么；
    
2. 训练数据是怎样从DataLoader传递到模型中的；
    
3. `direction=AtoB` 参数如何控制图像转换方向；
    
4. `real_A`、`real_B`与数据集A、B之间是什么关系；
    
5. `netG_A`和`netG_B`分别完成什么转换；
    
6. CycleGAN两个方向的图像转换是怎样实现的；
    
7. 项目中的基础模型类如何统一管理生成结果；
    
8. 预训练模型下载脚本是怎样工作的。
    

通过本周的学习，我开始从之前“理解一个GAN训练循环”逐渐过渡到“阅读一个完整GAN项目的代码组织和数据执行流程”。

---

## 二、CycleGAN项目整体结构的认识

本周首先对CycleGAN项目的整体目录结构进行了整理。

### 关键源码

```
pytorch-CycleGAN-and-pix2pix
│
├── train.py
├── test.py
│
├── models
│   ├── base_model.py
│   ├── cycle_gan_model.py
│   └── networks.py
│
├── data
│   └── ...
│
├── options
│   └── ...
│
├── scripts
│   └── download_cyclegan_model.sh
│
└── checkpoints
```

### 源码功能

CycleGAN项目将训练入口、数据读取、模型定义、网络结构、参数配置以及模型权重保存等功能分别放在不同文件和目录中。

### 自己的理解

之前学习DCGAN时，我主要关注生成器、判别器以及训练循环，因此容易认为GAN项目主要就是模型定义和反向传播代码。

阅读CycleGAN以后发现，一个完整的深度学习项目除了模型本身，还需要负责数据读取、参数管理、模型创建、训练测试、结果保存以及预训练权重加载等工作。

因此阅读这种开源项目时，不能一开始只盯着生成器和判别器，而应该先看清楚整个项目的目录结构以及不同文件之间的关系。

### 简单关系图

```
train.py / test.py
        │
        ├────────→ options
        │          读取运行参数
        │
        ├────────→ data
        │          创建数据集
        │
        └────────→ models
                   创建模型
                      │
                      ├── BaseModel
                      │
                      └── networks.py
                           │
                           ├── Generator
                           └── Discriminator
```

---

## 三、`get_current_visuals()`函数的理解

本周阅读 `BaseModel` 时，重点分析了 `get_current_visuals()` 方法。

### 关键源码

```python
def get_current_visuals(self):
    """Return visualization images."""
    visual_ret = OrderedDict()

    for name in self.visual_names:
        if isinstance(name, str):
            visual_ret[name] = getattr(self, name)

    return visual_ret
```

### 源码功能

根据 `self.visual_names` 中保存的变量名称，从当前模型对象中取得对应的图像数据，并统一保存到字典中返回。

### 自己的理解

最开始看到：

```
getattr(self, name)
```

时，我不太理解为什么不直接使用：

```
self.real_A
self.fake_B
```

后来发现，`name`本身是一个字符串。

例如：

```
name = "real_A"
```

那么：

```
getattr(self, name)
```

实际上相当于：

```
self.real_A
```

这样做以后，`BaseModel`不需要提前知道当前使用的是CycleGAN还是其他模型。

不同模型只需要定义自己的：

```
self.visual_names
```

BaseModel就能够使用同一套代码读取对应的图片结果。

### 简单关系图

```
self.visual_names
        │
        │
        ├── "real_A"
        ├── "fake_B"
        └── "rec_A"
        │
        ↓
      name
        ↓
getattr(self, name)
        ↓
取得模型中的对应变量
        ↓
visual_ret
        ↓
显示或保存结果
```

通过这段源码，我开始理解父类在完整PyTorch项目中的作用，它可以把不同模型都需要使用的公共功能统一封装起来。

---

## 四、`set_input()`函数的理解

本周重点阅读了 `CycleGANModel` 中的 `set_input()` 方法，这也是本周代码阅读过程中产生疑问比较多的一部分。

### 关键源码

```python
def set_input(self, input):
    """Unpack input data from the dataloader and
    perform necessary pre-processing steps."""

    AtoB = self.opt.direction == "AtoB"

    self.real_A = input["A" if AtoB else "B"].to(self.device)

    self.real_B = input["B" if AtoB else "A"].to(self.device)

    self.image_paths = input[
        "A_paths" if AtoB else "B_paths"
    ]
```

### 源码功能

从DataLoader传入的数据字典中取得A、B两个域的图片，并根据 `direction` 参数决定当前 `real_A` 和 `real_B` 分别对应哪一组输入数据。

### 自己的理解

这段代码中最开始不理解的是：

```
input["A" if AtoB else "B"]
```

将它拆开以后就比较容易理解。

首先：

```
AtoB = self.opt.direction == "AtoB"
```

是在判断当前设置的图像转换方向。

如果：

```
direction = AtoB
```

那么：

```
AtoB = True
```

最终得到：

```
self.real_A = input["A"]
self.real_B = input["B"]
```

如果：

```
direction = BtoA
```

那么：

```
AtoB = False
```

最终得到：

```
self.real_A = input["B"]
self.real_B = input["A"]
```

因此，`direction`参数实际上决定的是数据进入模型以后，A、B两个输入怎样对应到模型内部的 `real_A` 和 `real_B`。

### 简单流程图

```
             direction
                 │
        ┌────────┴────────┐
        │                 │
       AtoB              BtoA
        │                 │
        ↓                 ↓
real_A=input["A"]   real_A=input["B"]

real_B=input["B"]   real_B=input["A"]
```

这也让我认识到，阅读源码时不能仅仅根据 `real_A` 这个变量名称判断它到底是哪一种图片，还需要结合当前 `direction` 参数进行分析。

---

## 五、`real_A`与`real_B`的对应关系

在阅读 `set_input()` 时，我最开始容易直接把：

```
real_A = 马
real_B = 斑马
```

固定下来，但后来发现这种理解并不完全准确。

### 关键源码

```python
AtoB = self.opt.direction == "AtoB"

self.real_A = input["A" if AtoB else "B"].to(self.device)

self.real_B = input["B" if AtoB else "A"].to(self.device)
```

### 源码功能

根据当前运行方向，将DataLoader中的A域和B域图片分别赋给模型内部的 `real_A` 和 `real_B`。

### 自己的理解

假设原始数据集定义为：

```
A = horse
B = zebra
```

如果当前：

```
direction = AtoB
```

那么：

```
real_A = horse
real_B = zebra
```

但是当运行方向发生改变时，两组输入数据的对应关系也可能发生交换。

因此现在我更倾向于把：

```
real_A
real_B
```

理解成：

```
CycleGAN当前运行过程中定义的A域输入
CycleGAN当前运行过程中定义的B域输入
```

而不是永久将它们对应成某一种具体图片。

### 简单关系图

```
原始数据集

input["A"]          input["B"]
    │                   │
    │     direction     │
    └────────┬──────────┘
             ↓

         set_input()
             │
      ┌──────┴──────┐
      ↓             ↓
   real_A         real_B
```

这一部分解决了我阅读CycleGAN源码过程中关于A域、B域以及变量名称之间关系的一个主要疑问。

---

## 六、`netG_A`和`netG_B`转换方向的理解

CycleGAN与之前学习的DCGAN存在一个非常明显的区别，就是CycleGAN同时使用两个生成器。

### 关键源码

```python
self.fake_B = self.netG_A(self.real_A)

self.fake_A = self.netG_B(self.real_B)
```

### 源码功能

`netG_A`负责将A域图片转换到B域，`netG_B`负责将B域图片转换到A域。

### 自己的理解

可以简单记成：

```
netG_A：A → B

netG_B：B → A
```

例如在经典的horse2zebra任务中，如果定义：

```
A = horse
B = zebra
```

那么：

```
netG_A：horse → zebra

netG_B：zebra → horse
```

之前阅读代码时，我曾经产生过一个疑问：

如果 `real_A` 是斑马，而 `netG_A` 是马到斑马，那么把斑马输入 `netG_A` 是否不合理？

后来发现，这个问题不能只看某一个变量，而要结合：

```
数据集A/B的定义
+
direction参数
+
模型内部real_A/real_B的对应关系
```

一起分析。

### 简单关系图

```
                  netG_A
       A域 ─────────────────→ B域


       A域 ←───────────────── B域
                  netG_B
```

如果是horse2zebra任务：

```
                  netG_A
      horse ───────────────→ zebra


      horse ←─────────────── zebra
                  netG_B
```

这样整理以后，两个生成器分别负责什么方向就比较清楚了。

---

## 七、CycleGAN双向循环转换过程

在理解两个生成器以后，本周进一步整理了CycleGAN的基本前向转换过程。

### 关键源码

```python
def forward(self):
    self.fake_B = self.netG_A(self.real_A)

    self.rec_A = self.netG_B(self.fake_B)

    self.fake_A = self.netG_B(self.real_B)

    self.rec_B = self.netG_A(self.fake_A)
```

### 源码功能

分别完成A→B和B→A两个方向的图像生成，并将生成图片再次通过另一个生成器转换回原来的图像域。

### 自己的理解

第一条路径为：

```
self.fake_B = self.netG_A(self.real_A)
```

也就是：

```
real_A → fake_B
```

接下来：

```
self.rec_A = self.netG_B(self.fake_B)
```

再把生成的B域图片转换回A域：

```
fake_B → rec_A
```

完整过程就是：

```
real_A → fake_B → rec_A
```

另一个方向同样可以得到：

```
real_B → fake_A → rec_B
```

变量名称也可以这样理解：

```
real：真实输入图片

fake：第一次生成出来的图片

rec：reconstructed，重新转换回来的图片
```

### 简单流程图

```
A方向循环

real_A
   │
   ↓
 netG_A
   │
   ↓
fake_B
   │
   ↓
 netG_B
   │
   ↓
 rec_A
```

另一个方向：

```
B方向循环

real_B
   │
   ↓
 netG_B
   │
   ↓
fake_A
   │
   ↓
 netG_A
   │
   ↓
 rec_B
```

两个方向组合起来：

```
          netG_A
real_A ───────────→ fake_B
  ↑                    │
  │                    │
  └────── netG_B ──────┘
          得到rec_A


          netG_B
real_B ───────────→ fake_A
  ↑                    │
  │                    │
  └────── netG_A ──────┘
          得到rec_B
```

本周主要先理解这一转换过程，目前还没有继续深入分析Cycle Consistency Loss具体怎样约束 `real_A` 和 `rec_A`。

---

## 八、模型可视化结果之间的关系

结合前面阅读的 `get_current_visuals()`，本周还整理了CycleGAN中常见的几组图像变量。

### 关键源码

```
real_A
fake_B
rec_A

real_B
fake_A
rec_B
```

获取这些变量时使用：

```
visual_ret[name] = getattr(self, name)
```

### 源码功能

统一取得CycleGAN中的真实输入图像、生成结果和重建结果，供后续训练可视化或测试结果保存使用。

### 自己的理解

这六个变量看起来比较多，但实际上可以分成两组。

第一组：

```
real_A → fake_B → rec_A
```

第二组：

```
real_B → fake_A → rec_B
```

例如在horse2zebra任务中，可以把第一组简单理解成：

```
真实马
  ↓
生成斑马
  ↓
重新转换成马
```

因此后续实际运行模型时，将这三张图片放在一起观察，就可以比较直观地看出模型每一步做了什么。

### 简单关系图

```
┌────────── A循环 ──────────┐

real_A  →  fake_B  →  rec_A

└───────────────────────────┘


┌────────── B循环 ──────────┐

real_B  →  fake_A  →  rec_B

└───────────────────────────┘
```

---

## 九、预训练模型下载脚本的学习

本周除了Python源码以外，还阅读了项目中的：

```
download_cyclegan_model.sh
```

脚本。

### 关键源码

```shell
FILE=$1

echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained

MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
```

### 源码功能

接收运行脚本时传入的模型名称，创建对应的checkpoint目录，并确定预训练生成器权重的保存路径。

### 自己的理解

其中：

```
FILE=$1
```

表示取得运行脚本时传入的第一个参数。

例如执行：

```
bash ./scripts/download_cyclegan_model.sh horse2zebra
```

那么：

```
FILE = horse2zebra
```

接下来：

```
mkdir -p ./checkpoints/${FILE}_pretrained
```

就会创建：

```
checkpoints/horse2zebra_pretrained
```

目录。

模型权重最终会放在类似：

```
latest_net_G.pth
```

这样的文件中。

### 简单流程图

```
执行.sh脚本
      │
      ↓
传入horse2zebra
      │
      ↓
FILE=$1
      │
      ↓
FILE=horse2zebra
      │
      ↓
创建checkpoint目录
      │
      ↓
下载预训练模型
      │
      ↓
保存.pth权重
```

通过这一部分，我进一步理解了网络代码和模型权重之间的区别。

项目中的Python代码主要定义网络结构，而 `.pth` 文件中保存的是模型训练完成以后得到的参数。

---

## 十、本周CycleGAN代码执行流程整理

将本周阅读的几个关键部分连接起来以后，目前我对CycleGAN的基本数据流程有了比较清楚的认识。

### 关键源码

首先：

```
set_input(input)
```

得到：

```
real_A
real_B
```

然后：

```
fake_B = netG_A(real_A)

fake_A = netG_B(real_B)
```

再进一步得到：

```
rec_A = netG_B(fake_B)

rec_B = netG_A(fake_A)
```

最后：

```
get_current_visuals()
```

取得需要显示的生成结果。

### 源码功能

将数据读取、生成器转换、循环重建以及结果获取几个步骤连接起来，形成CycleGAN目前学习到的主要前向数据流。

### 自己的理解

把几个函数单独阅读时，很容易不知道它们为什么需要这样写。

现在把它们连接起来以后，可以分别理解为：

```
set_input()
解决数据从哪里进入模型
```

```
netG_A / netG_B
解决图片向哪个图像域转换
```

```
real / fake / rec
表示图片当前处于转换流程的哪个阶段
```

```
get_current_visuals()
解决怎样取得这些结果进行显示
```

### 本周整体流程图

```
             DataLoader
                 │
                 ↓
            input字典
                 │
                 ↓
          set_input(input)
                 │
          ┌──────┴──────┐
          ↓             ↓
       real_A          real_B
          │             │
          ↓             ↓
       netG_A          netG_B
          │             │
          ↓             ↓
       fake_B          fake_A
          │             │
          ↓             ↓
       netG_B          netG_A
          │             │
          ↓             ↓
        rec_A           rec_B
          │             │
          └──────┬──────┘
                 ↓
       get_current_visuals()
                 │
                 ↓
          查看或保存结果
```

通过这张图，目前已经能够把本周阅读过的主要变量和函数连接起来。

---

## 十一、本周学习收获

本周最大的收获是开始真正进入CycleGAN完整项目源码的阅读，而不是只停留在GAN基本原理和单独的训练循环上。

首先，通过阅读 `BaseModel` 中的 `get_current_visuals()`，我开始理解完整PyTorch项目为什么要使用父类统一管理公共功能。

其次，通过分析 `set_input()`，我理解了DataLoader中的A、B数据进入模型以后还会受到 `direction` 参数的影响，也进一步解决了 `real_A` 和 `real_B` 到底表示什么的问题。

另外，本周重点整理了：

```
netG_A：A → B

netG_B：B → A
```

两个生成器之间的转换关系，并能够理解：

```
real_A → fake_B → rec_A

real_B → fake_A → rec_B
```

两个基本循环。

相比上周主要分析：

```
zero_grad()
backward()
optimizer.step()
detach()
```

本周开始更多关注完整项目中的：

```
数据读取
+
函数调用
+
生成器方向
+
图片变量之间的关系
+
模型文件和权重管理
```

学习内容开始由“理解某几行GAN训练代码”逐渐向“阅读完整深度学习项目源码”过渡。

---

## 十二、目前仍存在的问题

虽然目前已经能够理解CycleGAN的基本代码结构和前向数据流，但还有一些内容没有深入学习。

第一，目前已经能够理解：

```
real_A → fake_B → rec_A
```

这一转换过程，但还没有深入分析为什么需要让 `rec_A` 与 `real_A` 保持接近。

第二，目前已经理解两个生成器：

```
netG_A
netG_B
```

的基本作用，但对于：

```
netD_A
netD_B
```

两个判别器分别判断什么内容，还需要结合后续训练代码继续学习。

第三，目前主要分析的是CycleGAN的前向数据流，还没有像之前学习DCGAN一样，对生成器和判别器的梯度传播以及参数更新顺序进行完整分析。

第四，目前还没有系统学习CycleGAN中的：

```
GAN Loss
Cycle Consistency Loss
Identity Loss
```

分别起什么作用。

因此，本周主要完成的是CycleGAN项目结构和基础数据流的第一阶段源码阅读。

---

## 十三、下一步学习安排

下一步准备继续按照CycleGAN真实的代码执行顺序进行学习。

首先继续分析生成器的前向传播过程，将：

```
real_A → fake_B → rec_A

real_B → fake_A → rec_B
```

与实际源码进一步对应。

之后开始阅读生成器损失计算部分，重点理解：

```
GAN Loss

Cycle Consistency Loss

Identity Loss
```

分别解决什么问题。

然后继续分析：

```
netD_A
netD_B
```

两个判别器的训练过程，并与之前DCGAN中已经学习过的：

```
真实图片
    ↓
判别器

生成图片
    ↓
detach()
    ↓
判别器

backward()
    ↓
optimizer.step()
```

进行对比。

最后再完整阅读CycleGAN中生成器和判别器的参数更新过程，尝试整理出一个batch的完整训练流程：

```
输入图片
   ↓
生成器前向传播
   ↓
计算生成器损失
   ↓
更新生成器
   ↓
计算判别器损失
   ↓
更新判别器
   ↓
进入下一batch
```

总体来看，本周已经初步看清了CycleGAN项目中数据从哪里进入、两个生成器怎样完成双向图像转换、`real`、`fake`、`rec`几个变量怎样对应，以及模型生成结果怎样统一获取。

下一阶段将继续进入CycleGAN损失函数和完整训练机制的学习。
