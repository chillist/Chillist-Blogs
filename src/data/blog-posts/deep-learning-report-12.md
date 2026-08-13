---
title: "DCGAN 梯度更新与 Transformer 因果掩码"
slug: deep-learning-report-12
publishDate: 2026-07-31
description: "分析 DCGAN 的权重初始化与梯度更新机制，并通过手动实现多头自注意力理解因果掩码。"
---
## 一、学习概述

本周继续进行深度学习相关代码的阅读和理解，学习内容主要分为两个部分。

第一部分是在上周DCGAN项目学习的基础上，进一步分析生成器和判别器的训练代码。上周已经对DCGAN的网络结构、MNIST训练过程、生成结果以及训练后期出现的模式坍塌和损失饱和现象进行了整理。本周没有继续单纯增加训练轮数，而是把重点转向训练循环中的具体代码，主要分析了权重初始化、判别器训练、生成器训练、`detach()`、反向传播和优化器更新之间的关系。

第二部分是重新学习Transformer中的注意力掩码，并尝试通过手动实现多头自注意力，理解下三角掩码、上三角区域以及因果注意力的实现方式。

通过本周学习，我对“计算损失”“计算梯度”和“更新参数”这三个过程之间的区别有了更加清楚的认识，也进一步理解了PyTorch自动求导机制在GAN和Transformer中的具体使用方法。

---

## 二、本周学习内容与上周的衔接

上周主要完成了以下工作：

1. 阅读DCGAN项目整体代码。
    
2. 整理生成器和判别器网络结构。
    
3. 分析MNIST前20轮生成结果。
    
4. 观察第14轮以后出现的生成结果退化。
    
5. 结合损失和生成图片分析模式坍塌与训练失衡。
    

上周虽然已经能够说明DCGAN的整体训练过程，但对部分代码仍然存在疑问，例如：

- 为什么生成器和判别器需要分别初始化权重？
    
- 为什么判别器需要对真实图像和生成图像分别反向传播？
    
- `fake.detach()`到底阻断了什么？
    
- 生成器更新时仍然调用了判别器，为什么最终只更新生成器？
    
- `zero_grad()`、`backward()`和`step()`分别负责什么？
    
- `D(x)`、`D(G(z))`这些数值应当怎样理解？
    

因此，本周主要围绕这些问题进行进一步学习。

---

## 三、DCGAN权重初始化的理解

DCGAN在创建生成器和判别器后，会调用权重初始化函数：

``` python
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
```

这个函数会被应用到网络中的每一个子模块。

### 1. 卷积层初始化

对于普通卷积层和转置卷积层，权重使用均值为0、标准差为0.02的正态分布初始化：

```
torch.nn.init.normal_(m.weight, 0.0, 0.02)
```

这里不是把所有卷积权重初始化为0，而是让权重集中在0附近，同时保留较小的随机差异。

如果所有权重都完全相同，不同神经元在训练时可能学习到相同的特征。使用随机初始化能够破坏这种对称性，使不同卷积核学习不同的图像特征。

### 2. 批归一化层初始化

对于BatchNorm层，缩放参数以1为中心进行初始化，偏置初始化为0：

```
torch.nn.init.normal_(m.weight, 1.0, 0.02)
torch.nn.init.zeros_(m.bias)
```

BatchNorm的基本输出可以写成：

```
输出 = γ × 标准化结果 + β
```

其中，`weight`相当于γ，`bias`相当于β。

将γ初始化到1附近、β初始化为0，可以让BatchNorm在训练刚开始时尽量保持输入特征的原始分布，同时再通过后续训练自动调整。

### 3. 初始化的重要性

GAN中存在生成器和判别器两个相互对抗的网络。如果其中一个网络在初始阶段明显强于另一个网络，就可能造成训练失衡。

因此，权重初始化虽然不能完全避免模式坍塌，但能够让两个网络从相对合理的状态开始训练，减少初始参数过大或过小带来的不稳定。

---

## 四、判别器对真实图像的训练过程

每个batch开始时，程序首先训练判别器识别真实图像。

``` python
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

D_x = output.mean().item()
```

### 1. 清空判别器梯度

```
netD.zero_grad()
```

PyTorch中的梯度默认采用累加方式保存。如果不清空梯度，新一轮计算出的梯度就会与上一轮梯度相加。

因此，在训练判别器之前，需要先清除判别器参数中上一轮残留的梯度。

### 2. 构造真实标签

```
label = torch.full((batch_size,), real_label)
```

由于真实图像的目标是1，因此该标签张量中的所有元素都为1。

假设当前批次有64张图像，则标签形状为：

```
[64]
```

判别器输出同样是每张图像对应一个真假概率，因此也会整理成`[64]`，两者可以直接计算二元交叉熵损失。

### 3. 计算真实图像损失

```
output = netD(real_cpu)
errD_real = criterion(output, label)
```

判别器接收真实图像，输出每张图像属于真实数据的概率。

如果判别器对真实图像的输出接近1，`errD_real`通常会比较小；如果输出接近0，损失就会比较大。

### 4. 第一次反向传播

```
errD_real.backward()
```

这一步根据真实图像损失计算判别器参数的梯度，但还没有立即更新参数。

此时完成的是：

```
真实图像
   ↓
判别器
   ↓
真实图像损失
   ↓
计算判别器梯度
```

### 5. D(x)的含义

```
D_x = output.mean().item()
```

`D_x`表示判别器对当前批次真实图像输出概率的平均值。

理想情况下，判别器应当将真实图像判断为真实，因此`D_x`应当相对接近1。但如果它长期精确等于1，也可能说明判别器过强或输出已经进入饱和状态。

---

## 五、判别器对生成图像的训练过程

完成真实图像的梯度计算后，程序继续让判别器识别生成图像。

``` python
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

D_G_z1 = output.mean().item()

errD = errD_real + errD_fake
optimizerD.step()
```

### 1. 生成随机噪声

```
noise = torch.randn(batch_size, nz, 1, 1)
```

假设：

```
batch_size = 64
nz = 100
```

则噪声张量形状为：

```
[64, 100, 1, 1]
```

每一组100维随机噪声经过生成器后，会得到一张生成图像。

### 2. 生成假图像

```
fake = netG(noise)
```

假设当前使用MNIST数据，则输出形状为：

```
[64, 1, 64, 64]
```

由于`fake`是通过`netG(noise)`计算得到的，所以它原本保留着从输出返回生成器参数的计算路径。

### 3. 修改标签

```
label.fill_(fake_label)
```

这里没有重新创建标签张量，而是把原有标签张量中的所有1修改为0。

此时判别器需要把生成图像判断为假，因此目标标签是0。

### 4. detach的作用

```
output = netD(fake.detach())
```

`fake.detach()`会得到一个与`fake`数值相同的新张量，但这个张量不再保留通向生成器的梯度计算路径。

可以把原始计算图理解为：

```
随机噪声
   ↓
生成器参数
   ↓
fake
   ↓
判别器
   ↓
损失
```

使用`detach()`以后，梯度传播会变成：

```
fake.detach()
   ↓
判别器
   ↓
损失
   ↓
只计算判别器梯度
```

因此，在训练判别器识别假图像时，不会顺便修改生成器。

### 5. 第二次反向传播

```
errD_fake.backward()
```

这一步会把识别假图像产生的梯度继续累加到判别器参数中。

此时判别器参数中保存的梯度相当于：

```
判别器总梯度
=
识别真实图像的梯度
+
识别生成图像的梯度
```

### 6. 更新判别器

```
optimizerD.step()
```

只有执行`optimizerD.step()`以后，判别器参数才真正发生变化。

因此，判别器训练可以总结为：

```
清空判别器梯度
        ↓
真实图像前向传播
        ↓
真实图像损失反向传播
        ↓
生成图像前向传播
        ↓
假图像损失反向传播
        ↓
两部分梯度累加
        ↓
optimizerD.step()
        ↓
更新判别器参数
```

---

## 六、生成器更新过程

判别器更新完成后，程序开始训练生成器。

``` python
netG.zero_grad()

label.fill_(real_label)

output = netD(fake)
errG = criterion(output, label)

errG.backward()

D_G_z2 = output.mean().item()

optimizerG.step()
```

### 1. 清空生成器梯度

```
netG.zero_grad()
```

这一步会清除生成器上一轮训练残留的梯度。

### 2. 将生成图像的目标设置为真实

```
label.fill_(real_label)
```

虽然输入判别器的仍然是生成图像，但训练生成器时，标签却被设置为1。

原因是生成器的目标不是正确判断真假，而是让判别器错误地把生成图像判断为真实。

也就是说，生成器希望：

```
D(G(z)) → 1
```

### 3. 不再使用detach

```
output = netD(fake)
```

这里使用的是原始`fake`，没有调用`detach()`。

因此，损失可以经过判别器继续向前传播到生成器：

```
生成器参数
   ↓
fake
   ↓
判别器
   ↓
生成器损失
   ↓
梯度返回生成器
```

### 4. 计算生成器损失

```
errG = criterion(output, label)
```

当前标签是1。如果判别器认为生成图像是真实图像，生成器损失会下降；如果判别器认为生成图像是假的，生成器损失就会上升。

### 5. 生成器反向传播

```
errG.backward()
```

由于没有使用`detach()`，这次反向传播可以计算生成器参数的梯度。

### 6. 更新生成器参数

```
optimizerG.step()
```

`optimizerG`在创建时只接收了生成器参数：

```
optimizerG = optim.Adam(netG.parameters(), ...)
```

因此，执行`optimizerG.step()`时，只会读取并更新`netG.parameters()`中的参数。

---

## 七、为什么生成器训练时能够只更新生成器

本周重点理解了一个问题：

> 生成器训练时仍然调用了判别器，为什么最终能够只更新生成器参数？

原因主要不是网络名称，而是由优化器保存的参数范围决定的。

生成器优化器定义为：

```
optimizerG = optim.Adam(netG.parameters(), ...)
```

判别器优化器定义为：

```
optimizerD = optim.Adam(netD.parameters(), ...)
```

因此：

```
optimizerD.step()
```

只更新判别器参数。

```
optimizerG.step()
```

只更新生成器参数。

在生成器反向传播过程中，因为损失经过了判别器，所以PyTorch实际上也可能为判别器参数计算梯度。但是，只要不执行`optimizerD.step()`，这些判别器梯度就不会被用于参数更新。

在下一批判别器训练开始时：

```
netD.zero_grad()
```

又会把这些梯度清除。

所以，判断某一组参数是否真正发生变化，需要同时观察三个方面：

1. 该参数是否参与了前向传播。
    
2. 该参数是否计算了梯度。
    
3. 对应的优化器是否执行了`step()`。
    

其中，第三点直接决定参数是否真正更新。

为了减少无用的梯度计算，也可以在训练生成器时临时冻结判别器：

``` python
for param in netD.parameters():
    param.requires_grad_(False)

output = netD(fake)
errG = criterion(output, label)
errG.backward()
optimizerG.step()

for param in netD.parameters():
    param.requires_grad_(True)
```

这种写法不是基本训练流程必须使用的，但可以减少判别器参数的无效梯度计算。

---

## 八、zero_grad、backward和step的区别

通过GAN训练代码，本周进一步区分了PyTorch训练中的三个重要操作。

### 1. zero_grad

```
optimizer.zero_grad()
```

或者：

```
net.zero_grad()
```

作用是清除参数中已经保存的梯度。

它不会进行前向传播，也不会更新模型参数。

### 2. backward

```
loss.backward()
```

作用是根据当前损失和计算图，计算各个参数的梯度。

它只负责计算或累加梯度，不会直接修改参数值。

### 3. step

```
optimizer.step()
```

作用是由优化器读取参数中的梯度，并按照Adam或SGD等优化规则修改参数。

三者之间的关系为：

```
zero_grad()
清除旧梯度
      ↓
前向传播
计算模型输出
      ↓
计算损失
      ↓
backward()
计算新梯度
      ↓
step()
根据梯度更新参数
```

之前容易把`backward()`理解为“模型已经完成更新”。本周学习后认识到，`backward()`只是计算梯度，真正更新参数的是`optimizer.step()`。

---

## 九、D(G(z))两个数值的理解

DCGAN训练日志中通常会打印两个`D(G(z))`：

```
D_G_z1 = output.mean().item()
```

和：

```
D_G_z2 = output.mean().item()
```

第一个值来自判别器训练阶段：

```
output = netD(fake.detach())
```

它表示判别器在识别生成图像时，对生成图像属于真实数据的平均判断。

第二个值来自生成器训练阶段：

```
output = netD(fake)
```

它表示生成器计算自身损失时，判别器对同一批生成图像的平均输出。

需要注意的是，`D_G_z2`是在执行：

```
optimizerG.step()
```

之前记录的。

因此，它并不是“生成器参数已经更新完成以后重新生成图像得到的结果”，而是生成器本次优化所依据的判别器输出。

如果希望准确比较生成器更新前后的效果，需要在`optimizerG.step()`后重新执行一次生成器和判别器前向传播。

---

## 十、GAN训练流程的完整整理

本周将一个batch中的完整训练流程重新整理如下：

```
第一阶段：训练判别器
────────────────────────

清空判别器梯度
        ↓
输入真实图像
        ↓
判别器判断真实图像
        ↓
计算真实图像损失
        ↓
反向传播真实图像梯度
        ↓
随机噪声进入生成器
        ↓
得到生成图像
        ↓
fake.detach()
        ↓
判别器判断生成图像
        ↓
计算假图像损失
        ↓
反向传播假图像梯度
        ↓
optimizerD.step()
        ↓
更新判别器


第二阶段：训练生成器
────────────────────────

清空生成器梯度
        ↓
继续使用当前生成图像
        ↓
不使用detach
        ↓
经过判别器得到真假概率
        ↓
把目标标签设置为真实
        ↓
计算生成器损失
        ↓
梯度经过判别器返回生成器
        ↓
optimizerG.step()
        ↓
更新生成器
```

通过这一流程可以看出，生成器和判别器不是同时更新，而是在同一个batch中依次进行两次优化。

---

## 十一、手动注意力模块的学习

在完成DCGAN训练代码的进一步分析后，本周还开始阅读手动实现的多头自注意力模块。

注意力模块输入通常为：

```
batch_size, seq_len, dim = x.shape
```

例如：

```
x.shape = [B, T, D]
```

其中：

- `B`表示批次数量。
    
- `T`表示序列长度。
    
- `D`表示每个token的特征维度。
    

通过三个线性层可以得到：

``` python
q = self.wq(x)
k = self.wk(x)
v = self.wv(x)
```

它们的初始形状均为：

```
[B, T, D]
```

之后再拆分成多个注意力头：

```
[B, T, D]
    ↓
[B, T, H, Dh]
    ↓
[B, H, T, Dh]
```

其中：

```
Dh = D / H
```

每个注意力头都会单独计算token之间的相关性。

---

## 十二、注意力分数矩阵

注意力分数通过Q和K的矩阵乘法得到：

``` python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(self.head_dim)
```

假设Q和K的形状都是：

```
[B, H, T, Dh]
```

将K最后两个维度交换后：

```
Kᵀ = [B, H, Dh, T]
```

矩阵相乘后得到：

```
scores = [B, H, T, T]
```

最后两个维度中的：

- 第一维T表示当前正在查询的token。
    
- 第二维T表示可以被关注的token。
    

矩阵中第`i`行第`j`列的元素，表示第`i`个token对第`j`个token的注意力分数。

---

## 十三、下三角因果掩码的理解

在自回归生成任务中，当前位置不能看到未来位置。

例如，生成第3个token时，只能使用第1、2、3个位置的信息，不能看到第4、5个位置。

因此，需要构造一个下三角矩阵：

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

可以通过下面的代码生成：

``` python
mask = torch.tril(
    torch.ones(
        seq_len,
        seq_len,
        device=x.device,
        dtype=torch.bool
    )
)
```

其中：

``` python
torch.tril(...)
```

会保留主对角线以及主对角线以下的区域。

然后把不能访问的位置填成负无穷：

``` python
scores = scores.masked_fill(~mask, float("-inf"))
```

经过Softmax以后：

``` python
attn = torch.softmax(scores, dim=-1)
```

被填成负无穷的位置会变成接近0的注意力权重。

因此，下三角掩码的作用可以表示为：

```
当前位置
   ↓
只能注意自己和之前的token
   ↓
未来token的分数设为负无穷
   ↓
Softmax后未来位置权重为0
```

---

## 十四、下三角掩码与上三角掩码的关系

在不同代码中，可能会看到下三角矩阵，也可能会看到上三角矩阵。

例如：

``` python
allowed_mask = torch.tril(torch.ones(T, T)).bool()
scores = scores.masked_fill(~allowed_mask, float("-inf"))
```

这里的下三角区域表示“允许访问的位置”。

另一种写法是：

``` python
future_mask = torch.triu(
    torch.ones(T, T),
    diagonal=1
).bool()

scores = scores.masked_fill(future_mask, float("-inf"))
```

这里的上三角区域表示“需要屏蔽的未来位置”。

两种方式的最终效果相同：

```
下三角：记录哪些位置可以看
上三角：记录哪些位置不能看
```

关键不是使用了`tril`还是`triu`，而是要结合`masked_fill`的条件判断，确认布尔值为`True`的位置究竟表示保留还是屏蔽。

---

## 十五、本周学习收获

本周最大的收获是进一步区分了前向传播、梯度计算和参数更新之间的关系。

在判别器训练中，真实图像损失和生成图像损失会分别执行一次`backward()`，两部分梯度累加后，再通过一次`optimizerD.step()`更新判别器。

通过分析`fake.detach()`，我认识到它并不是复制一张新的生成图片，也不是禁止判别器使用这张图片，而是切断生成图像与生成器参数之间的梯度传播路径。

在生成器训练中，虽然前向传播仍然经过判别器，但最终执行的是`optimizerG.step()`，因此真正更新的是生成器参数。优化器所管理的参数范围决定了`step()`会修改哪些参数。

通过分析`D(x)`、`D(G(z))`、`Loss_D`和`Loss_G`，我也认识到GAN训练日志中的单个数值不能单独判断模型效果，需要结合生成图片和多个指标共同观察。

在注意力机制方面，本周进一步理解了注意力分数矩阵为什么是`T×T`，以及下三角掩码和上三角掩码在代码中的不同表示方法。

---

## 十六、目前仍存在的问题

虽然已经能够说明DCGAN训练代码的主要流程，但仍有一些内容需要继续学习。

第一，目前主要理解了经典DCGAN中的`Sigmoid + BCELoss`训练方式，还没有深入比较`BCEWithLogitsLoss`、WGAN损失和其他GAN训练目标之间的区别。

第二，虽然已经理解生成器更新时判别器不会被对应优化器修改，但还需要进一步比较“正常计算判别器梯度”和“临时冻结判别器参数”在运行效率上的差别。

第三，目前对注意力掩码的理解主要集中在单个因果掩码，还需要继续分析批次维度、注意力头维度以及Padding Mask和Causal Mask同时存在时的广播过程。

第四，手动注意力模块目前主要用于打印张量形状和理解执行过程，还没有在实际序列任务中进行完整训练和结果验证。

---

## 十七、下一步学习安排

下一步准备继续完善手动多头注意力模块，重点打印Q、K、V拆分前后的形状，以及注意力分数、掩码和Softmax输出的变化。

在掩码部分，将进一步整理以下三种情况：

1. Encoder中的Padding Mask。
    
2. Decoder中的Causal Mask。
    
3. Padding Mask与Causal Mask组合使用。
    

在DCGAN方面，后续准备在现有代码中加入判别器冻结操作，对比冻结前后的训练速度和显存占用。

同时计划修改训练日志保存方式，将每个batch中的以下指标保存到CSV文件：

``` python
Loss_D
Loss_G
D(x)
D(G(z)) before G training
D(G(z)) used for G training
```

之后再根据日志绘制损失曲线，并结合固定噪声生成结果，分析GAN训练状态。

总体来看，本周的学习重点从“知道代码执行了什么”，逐渐转向“理解梯度为什么沿这条路径传播、参数为什么只由对应优化器更新”。这为后续继续学习更复杂的GAN模型和Transformer结构打下了基础。
