---
title: "RWKV：融合 Transformer 并行训练与 RNN 高效推理"
slug: deep-learning-report-06
publishDate: 2026-06-26
description: "学习 RWKV 的整体架构、Time Mix、Channel Mix 与 WKV 状态，并对比 Transformer 和 Mamba。"
---
## 一、学习概述

本周主要围绕 RWKV 模型展开学习，同时结合前面已经学习过的 Transformer 和 Mamba，对三种序列模型的结构和区别进行了整理。前面学习 Transformer 时，重点理解了 Q、K、V、自注意力、多头注意力、Mask、Encoder、Decoder 以及代码实现。本周学习 RWKV，主要是想弄清楚它为什么既像 Transformer，又像 RNN。

通过阅读 RWKV 官网内容和相关结构图，本周初步了解了 RWKV 的基本思想。RWKV 可以理解为一种结合 RNN 和 Transformer 优点的序列模型。它希望训练时尽量像 Transformer 一样可以并行，推理时又像 RNN 一样只维护一个状态，不需要像 Transformer 那样保存越来越长的 KV Cache。

本周学习重点包括 RWKV 的整体架构、Time Mix、Channel Mix、WKV 机制、state 的作用，以及 RWKV、Transformer、Mamba 三种模型的对比。通过这一周的学习，对“注意力机制”和“状态递推”这两种处理历史信息的方式有了更清楚的认识。

---

## 二、RWKV 模型基本认识

RWKV 是一种用于处理序列数据的模型。序列数据就是有先后顺序的数据，例如文本、语音、时间序列等。对于语言模型来说，模型要根据前面的 token 预测后面的 token，所以如何保存和使用历史信息非常重要。

RWKV 这个名字可以拆开理解：

```text
R：Receptance，接收门，用来控制当前 token 接收多少信息
W：Weight，时间权重或衰减权重，用来控制历史信息如何保留
K：Key，键，用来表示当前信息的重要程度
V：Value，值，表示真正被写入和传递的信息内容
```

RWKV 中有类似 Transformer 中 K、V 的概念，但它不是标准 Transformer Attention。Transformer 会显式计算当前 token 和历史 token 之间的关系，而 RWKV 更像是把历史信息不断压缩进 state 中，后面的 token 再从 state 中读取需要的信息。

简单来说，Transformer 更像是每次都回头查看所有历史内容，而 RWKV 更像是边读边更新笔记，只保留一份不断更新的历史记忆。

---

## 三、RWKV 模型整体架构

根据 RWKV 官网中的模型结构图，RWKV 的整体流程可以理解为：

```text
Input Token
↓
Input Embedding
↓
RWKV Block × L
    ├─ Time Mix
    └─ Channel Mix
↓
LayerNorm
↓
Output Linear
↓
Softmax
↓
Next Token Probability
```

![学习报告 6 配图](/assets/blog/deep-learning/report-06/image-01.png)

输入 token 首先会经过 Input Embedding。Embedding 的作用是把 token id 转换成向量，因为模型不能直接处理文字，只能处理数字向量。

然后输入会经过很多层 RWKV Block。每个 RWKV Block 里面主要有两个部分：Time Mix 和 Channel Mix。Time Mix 负责处理历史信息，Channel Mix 负责处理当前 token 向量内部的特征变化。

经过多层 RWKV Block 之后，模型会通过 LayerNorm 进行归一化，再通过输出层映射到词表大小，最后用 Softmax 得到下一个 token 的概率分布。

这一部分可以简单理解为：

```text
Embedding：把文字变成向量
Time Mix：处理历史信息
Channel Mix：处理特征变化
state：保存历史记忆
Softmax：输出下一个 token 的概率
```

---

## 四、Time Mix 模块学习

Time Mix 是 RWKV 中非常重要的部分。这里的 Time 不是现实中的时间，而是序列中的位置。比如句子：

```text
我 喜欢 打 篮球
```

当模型处理到“篮球”时，前面的“我、喜欢、打”就是历史位置。Time Mix 要解决的问题就是：当前 token 应该如何利用前面的历史信息。

RWKV 的 Time Mix 会把当前 token 的表示和上一时刻的表示进行混合。这样模型在处理当前 token 时，不只是看当前 token 本身，也会参考上一个 token 以及之前递推下来的状态。

Time Mix 可以简单理解为：

```text
当前 token
+
上一时刻的信息
+
历史 state
↓
融合历史信息
↓
输出当前 token 的新表示
↓
更新 state
```

在 Time Mix 中，R、W、K、V 都参与了历史信息的处理。R 像一个门，控制当前 token 接收多少信息；W 控制历史信息随着距离变远如何衰减；K 表示当前信息的重要程度；V 表示真正要写入和传递的内容。

所以 Time Mix 的核心作用就是让 RWKV 能够记住前文，并且根据当前 token 读取合适的历史信息。

---

## 五、Channel Mix 模块学习

Channel Mix 是 RWKV Block 中的另一部分。这里的 Channel 可以理解为特征维度。比如一个 token 向量有 768 维，那么这些维度就可以看成不同的 channel。

Time Mix 主要负责处理不同 token 之间的历史关系，而 Channel Mix 主要负责处理同一个 token 内部不同特征维度之间的关系。它的作用和 Transformer 中的前馈神经网络比较像，都是为了增强模型对当前 token 表示的处理能力。

可以简单理解为：

```text
Time Mix：负责看历史
Channel Mix：负责加工当前 token 的特征
```

一个 RWKV Block 可以理解为：

```text
输入 x
↓
LayerNorm
↓
Time Mix
↓
残差连接
↓
LayerNorm
↓
Channel Mix
↓
残差连接
↓
输出
```

其中，残差连接和 Transformer 中的残差连接作用类似，都是为了让深层网络更容易训练，也能减少信息在多层传递过程中的损失。

---

## 六、RWKV 中 state 的理解

本周还重点学习了 RWKV 中的 state。state 可以理解为模型内部保存的历史信息，也可以看成一种隐藏状态。它不是最终输出，也不是模型长期保存的参数，而是在推理过程中随着 token 输入不断更新的中间状态。

普通 RNN 推理时会保存隐藏状态，当前输入会和上一步状态一起决定新的状态。RWKV 的推理方式也有类似特点：

```text
当前 token
+
上一时刻 state
↓
得到当前输出
+
更新后的 state
```

不过 RWKV 的 state 不只是一个简单的向量，而是一组分层保存的状态。每一层 RWKV Block 都会维护自己的 state，用来保存历史信息。

这点和 Transformer 不一样。Transformer 推理时通常需要保存历史 token 的 K、V，也就是 KV Cache。生成内容越长，KV Cache 越大。RWKV 则主要维护固定大小的 state，不需要把所有历史 token 的 K、V 都保存下来。

可以这样理解：

```text
Transformer：保存历史 K、V
RWKV：保存不断更新的 state
```

所以 RWKV 在长序列推理时，缓存压力相对更稳定。

---

## 七、WKV 机制理解

WKV 是 RWKV 中处理历史信息的关键机制。它可以理解成一种递推式的信息融合方式。

Transformer 的 Attention 更像是：

```text
当前 token 回头看所有历史 token
再决定关注谁
```

RWKV 的 WKV 更像是：

```text
历史信息不断写进 state
当前 token 从 state 中读取需要的信息
```

所以 RWKV 并不是没有利用历史信息，而是利用历史信息的方式不同。它不通过完整注意力矩阵来计算所有 token 之间的关系，而是通过 state 把历史信息递推地保存下来。

WKV 中几个部分可以这样理解：

```text
K：当前信息有多重要
V：当前信息具体是什么
W：历史信息怎么衰减
R：当前 token 接收多少历史信息
state：保存历史记忆
```

这也是 RWKV 和 Transformer 的核心区别之一。Transformer 更偏向显式地计算 token 关系，RWKV 更偏向把历史压缩到状态里。

---

## 八、Transformer 模型回顾

为了更好理解 RWKV，本周也回顾了 Transformer 的核心思想。Transformer 的核心是 Attention，也就是注意力机制。每个 token 会生成 Q、K、V。Q 用来查询，K 用来匹配，V 用来提供真正的信息内容。

Transformer 的优势是表达能力强，能够直接计算 token 之间的关系，特别适合复杂上下文建模。比如在一句很长的话中，当前 token 可以直接和前面很远位置的 token 建立联系。

但 Transformer 在推理时通常需要保存历史 token 的 K、V，也就是 KV Cache。这样可以避免每次生成新 token 时重复计算历史信息。不过随着生成内容越来越长，KV Cache 也会越来越大。

所以 Transformer 可以理解为：

```text
训练并行能力强
上下文建模能力强
推理时需要维护 KV Cache
长序列下显存压力较大
```

---

## 九、Mamba 模型回顾

Mamba 是一种基于状态空间模型的序列建模架构。它和 Transformer 的思路不同，不是通过完整注意力矩阵保存历史信息，而是通过 state 保存历史。

Mamba 的核心是选择性状态空间模型。普通状态空间模型会用隐藏状态保存历史信息，但更新方式相对固定。Mamba 的改进在于，它会根据当前输入动态决定怎么更新状态、怎么写入信息、怎么读取信息。

可以简单理解为：

```text
当前 token 进入模型
↓
模型判断哪些信息该记住
↓
更新 state
↓
从 state 中读取信息
↓
输出当前结果
```

Mamba 和 RWKV 都使用 state 保存历史信息，但两者的机制不同。Mamba 重点在于选择性状态空间模型，通过动态控制状态更新来处理序列；RWKV 重点在于 Time Mix 和 WKV 机制，通过递推 state 来融合历史信息。

![学习报告 6 配图](/assets/blog/deep-learning/report-06/image-02.png)

---

## 十、RWKV、Transformer 和 Mamba 的对比

通过本周学习，可以把三种模型的区别整理如下：

|对比项|Transformer|Mamba|RWKV|
|---|---|---|---|
|核心机制|Attention 注意力机制|选择性状态空间模型|Time Mix + WKV state|
|历史信息保存方式|推理时缓存历史 K、V|用 state 保存历史|用 state 保存历史|
|是否依赖完整注意力矩阵|是|否|否|
|推理缓存特点|KV Cache 随序列变长|state 大小相对固定|state 大小相对固定|
|当前 token 如何利用历史|当前 token 显式关注历史 token|当前 token 控制 state 的更新和读取|当前 token 从递推 state 中读取历史|
|训练特点|并行能力强|通过专门算法提高效率|尽量保持类似 Transformer 的并行训练|
|推理特点|逐 token 生成，并维护 KV Cache|逐 token 更新 state|逐 token 更新 state|
|优势|上下文建模能力强，结构成熟|长序列效率较好|推理省缓存，结合 RNN 和 Transformer 思路|
|需要重点理解|Q、K、V、Attention、Mask|state、选择性更新|Time Mix、Channel Mix、WKV、state|

从整体上看，Transformer 更偏向“显式关注历史”，Mamba 和 RWKV 更偏向“把历史压缩进状态”。Transformer 的优势是模型结构成熟，表达能力强；Mamba 的优势是通过选择性状态空间模型提高长序列效率；RWKV 的优势是用 RNN 式 state 推理，同时尽量保留 Transformer 式训练并行能力。

更直观地说：

```text
Transformer：每一步都能回头看历史 token
Mamba：把历史存在 state 里，当前 token 决定怎么更新和读取
RWKV：把历史递推进 state 里，用 Time Mix 和 WKV 融合历史
```

---

## 十一、本周学习收获

通过本周学习，我对 RWKV 的整体结构有了初步理解。RWKV 不是普通 Transformer，也不是传统 RNN，而是一种把 RNN 状态递推思想和 Transformer 中 K、V、门控、通道混合等思想结合起来的模型。它的关键不是构造完整注意力矩阵，而是通过 Time Mix 和 WKV state 保存历史信息。

本周另一个收获是进一步分清了模型参数、中间结果和推理状态之间的区别。Transformer 中，模型长期保存的是各种权重参数，Q、K、V 是当前输入临时生成的中间结果，推理时会缓存历史 K、V。Mamba 中，模型长期保存的是状态空间相关参数和生成动态中间量的权重，推理时主要维护 state。RWKV 中，Time Mix、Channel Mix 等权重是模型参数，state 是推理时维护的历史记忆。

通过把 RWKV、Transformer、Mamba 放在一起比较，可以看到当前序列模型的发展方向并不只有 Attention 一条路线。Transformer 依靠注意力机制显式建模 token 关系，Mamba 通过选择性状态空间模型建模长序列，RWKV 则通过递推 state 进行历史信息融合。三种模型都在解决同一个问题：如何让当前 token 更好地利用历史上下文，只是实现方式不同。

---

## 十二、后续学习安排

后续将继续围绕 RWKV 的结构细节展开学习，重点理解 Time Mix 中 R、W、K、V 的具体作用，以及 WKV state 是如何递推更新的。同时继续复习 Transformer 中 Q、K、V 和 KV Cache 的实现过程，避免把模型参数、中间激活值和推理缓存混在一起。
