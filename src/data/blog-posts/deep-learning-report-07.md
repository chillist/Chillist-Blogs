---
title: "RWKV 状态递推机制与 Transformer、Mamba 对比"
slug: deep-learning-report-07
publishDate: 2026-06-26
description: "深入梳理 RWKV 的 R、W、K、V、状态递推和核心模块，比较三类序列模型处理历史信息的方式。"
---
## 一、学习概述

本周主要围绕 RWKV 模型展开学习，并结合前面已经学习过的 Transformer 和 Mamba，对三种序列模型的历史信息处理方式进行了对比。本周学习的重点不再放在复杂公式推导上，而是先从整体结构和核心模块入手，理解 RWKV 为什么既有 RNN 的特点，又和 Transformer 有一定联系。

在前面学习 Transformer 时，重点理解了 Q、K、V、自注意力机制、多头注意力、Mask、Encoder、Decoder 以及 KV Cache 的作用。学习 Mamba 时，重点理解了状态空间模型、选择性状态更新、`Δ_t、B_t、C_t` 以及 state 的作用。本周学习 RWKV 时，主要围绕 Time Mix、Channel Mix、R/W/K/V 机制、WKV state 的递推更新方式展开。

通过本周学习，我对 RWKV 的整体思路有了初步认识。RWKV 不像 Transformer 那样依赖完整的注意力矩阵，也不是普通的 RNN。它更像是把 Transformer 中的 K、V、门控和特征变换思想，与 RNN 中逐步更新 hidden state 的思想结合起来。RWKV 在推理时主要维护 state，而不是像 Transformer 那样不断保存越来越长的 KV Cache，因此它在长序列推理中具有一定优势。

---

## 二、RWKV 模型基本认识

RWKV 是一种用于序列建模的模型。序列建模指的是处理有先后顺序的数据，例如文本、语音、时间序列等。在语言模型中，模型需要根据前面的 token 预测后面的 token，因此如何保存和利用历史信息非常关键。

RWKV 这个名字可以拆成四个部分：

```text
R：Receptance，接收门，用来控制当前 token 接收多少历史信息
W：Weight，时间权重或衰减权重，用来控制旧信息保留多久
K：Key，键，用来表示当前信息的重要程度或写入标记
V：Value，值，表示当前 token 真正提供的信息内容
```

从名字可以看出，RWKV 中保留了类似 Transformer 中 K、V 的概念。但 RWKV 和 Transformer 的信息处理方式并不一样。Transformer 会显式计算当前 token 和历史 token 之间的注意力关系，而 RWKV 更强调用 state 递推地保存历史信息。

可以简单理解为：

```text
Transformer：每次生成时都可以回头查看历史 token 的 K、V
RWKV：边读边把历史信息整理进 state，后续 token 再从 state 中读取需要的信息
```

所以 RWKV 的核心不是保存所有历史 token，而是不断更新一个内部状态。这个 state 可以理解为模型内部的历史记忆。

---

## 三、RWKV 模型整体架构

RWKV 的整体结构可以简化理解为：

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

【图1 RWKV 模型整体架构图】  
此处可以插入 RWKV 官网中的模型架构图，用于展示 Input Embedding、RWKV Block、Time Mix、Channel Mix、LayerNorm、Output Linear 和 Softmax 的整体流程。

输入文本首先会被转换成 token id，然后通过 Input Embedding 转换成向量表示。模型不能直接处理文字本身，只能处理数字向量，因此 Embedding 是文本进入模型的第一步。

接下来，输入向量会经过多层 RWKV Block。每个 RWKV Block 主要包括两个部分：Time Mix 和 Channel Mix。

Time Mix 负责处理时间维度上的历史信息。这里的时间不是现实中的秒，而是序列中的位置。例如一句话中，当前 token 前面的所有 token 都可以看成历史位置。Time Mix 的作用就是让当前 token 利用前面的信息。

Channel Mix 负责处理特征维度上的信息。一个 token 向量通常有很多维度，这些维度可以理解为不同的特征通道。Channel Mix 的作用类似 Transformer 中的前馈神经网络，用来增强当前 token 表示的表达能力。

经过多层 RWKV Block 后，模型会通过 LayerNorm 稳定数值分布，再通过输出层映射到词表大小，最后经过 Softmax 得到下一个 token 的概率分布。

---

## 四、Time Mix 模块学习

Time Mix 是 RWKV 中最重要的模块之一。它主要负责处理历史信息，也就是让当前 token 不只是看到自己，还能利用前面 token 留下来的信息。

比如输入句子：

```text
我 喜欢 打 篮球
```

当模型处理到“篮球”时，它不能只看“篮球”这个词，还需要知道前面出现过“我”“喜欢”“打”。这些前文信息会通过 state 逐步保存下来，Time Mix 就负责更新和读取这些历史信息。

Time Mix 可以理解成下面这个过程：

```text
当前 token 的表示
+
上一时刻 token 的表示
+
之前递推下来的 state
↓
融合历史信息
↓
得到当前 token 的新表示
↓
更新 state
```

这里有一个很重要的点：RWKV 不是把所有历史 token 都原封不动保存下来，而是通过 state 对历史信息进行压缩和整理。每来一个新的 token，模型都会根据当前 token 更新 state，同时从 state 中读取对当前有用的信息。

所以 Time Mix 的作用可以概括为：

```text
负责看历史
负责更新 state
负责让当前 token 融合前文信息
```

这和 Transformer 的 Attention 有相似之处，因为二者都在解决“当前 token 如何利用历史信息”的问题。但实现方式不同。Transformer 是显式计算注意力权重，RWKV 是通过 state 递推融合历史。

---

## 五、Time Mix 中 R、W、K、V 的具体作用

RWKV 中的 R、W、K、V 是理解 Time Mix 的关键。可以先用一句话记住：

```text
R 负责读，W 负责忘，K 负责标记，V 负责内容，state 负责记忆。
```

### 1. R：控制当前 token 读多少历史信息

R 是 Receptance，可以理解为接收门。它的作用是控制当前 token 要从历史 state 中接收多少信息。

如果 R 的值较大，可以理解为当前 token 需要更多历史信息；如果 R 的值较小，可以理解为当前 token 只需要少量历史信息。

例如处理“篮球”时，模型可能需要从 state 中读取“喜欢”“打”等历史信息，因为这些词和“篮球”的语义关系比较强。这时 R 就像一个阀门，控制当前 token 从历史 state 中读取多少内容。

可以简单理解为：

```text
R 大：多读取历史信息
R 小：少读取历史信息
```

### 2. W：控制旧信息保留多久

W 是 Weight，在 RWKV 中主要可以理解为时间衰减权重。它控制历史信息在 state 中保留多久，以及旧信息随着时间推移会衰减多少。

如果某些信息距离当前 token 很远，但仍然重要，模型需要让它在 state 中保留得久一些。如果某些信息不太重要，或者已经过时，就可以让它衰减得快一些。

可以把 W 理解成历史信息的“保质期”：

```text
W 控制旧信息是否继续保留
W 也控制旧信息被遗忘的速度
```

这也是 RWKV 能够用固定大小 state 保存历史信息的原因之一。它不会无限制保留所有细节，而是通过 W 对旧信息进行衰减和筛选。

### 3. K：决定当前信息如何写入 state

K 是 Key，可以理解为当前 token 信息的标记。它决定当前 token 的信息以什么方式影响 state。

在 Transformer 中，K 通常用于和 Q 计算匹配关系；在 RWKV 中，K 更多参与当前信息写入 state 的过程。

可以简单理解为：

```text
K 决定当前信息怎么被标记
K 也影响当前信息写入 state 的方式
```

比如当前 token 是“篮球”，它携带的是一个和运动、名词、动作对象相关的信息。K 就像是给这条信息打标签，告诉 state 这条信息该如何被组织和保存。

### 4. V：当前 token 真正提供的内容

V 是 Value，表示当前 token 真正提供的信息内容。

如果说 K 更像是“标签”或“写入方式”，那么 V 就是“具体内容”。当前 token 中真正要写入 state 的语义信息，主要由 V 表示。

比如读到“篮球”时：

```text
K：告诉模型这条信息怎么写入 state
V：表示“篮球”这个 token 的具体语义内容
```

所以 K 和 V 通常是配合工作的：

```text
K 决定怎么写
V 决定写什么
```

---

## 六、WKV state 是什么

WKV state 可以理解为 RWKV 内部保存历史信息的状态。它不是模型最终输出，也不是模型长期保存的权重参数，而是在推理过程中不断更新的中间状态。

这点要和 Transformer 的 KV Cache 区分开。

Transformer 推理时通常会保存历史 token 的 K、V：

```text
K_cache = [K1, K2, K3, ..., Kt]
V_cache = [V1, V2, V3, ..., Vt]
```

随着生成的 token 越来越多，KV Cache 也会越来越大。

RWKV 不这样保存所有历史 K、V。它更像是维护一个不断更新的 state：

```text
state_1
state_2
state_3
...
```

每读一个新的 token，模型就把新信息写进 state，同时让旧信息按照 W 进行衰减。这样模型不用保存所有历史 token 的细节，而是把历史内容压缩进一个状态里。

可以把 WKV state 理解成一本不断更新的笔记：

```text
旧笔记不会全部丢掉，但会有些内容逐渐变淡；
新 token 会把新的重点信息写进笔记；
当前 token 需要信息时，就从这本笔记里读。
```

这本“笔记”就是 state。

---

## 七、WKV state 的递推更新过程

WKV state 的递推更新可以用人话理解为四步：

```text
旧信息先衰减
新信息再写入
得到新的 state
当前 token 从 state 中读取信息
```

更具体一点：

第一步，旧 state 会先经过 W 控制的衰减。  
这一步的作用是让旧信息不会无限制保留。比较重要的信息可能保留得久一些，不重要的信息会逐渐减弱。

第二步，当前 token 会生成 K 和 V。  
K 决定当前信息如何写入 state，V 表示当前 token 真正提供的信息内容。

第三步，模型把当前 token 的新信息写入 state。  
这样 state 就从上一步的历史记忆，更新成包含当前 token 信息的新记忆。

第四步，当前 token 通过 R 从 state 中读取信息。  
R 控制读取多少，最后得到当前 token 融合历史后的表示。

可以用伪代码理解这个流程：

```python
for each token:
    old_state = state

    # 旧信息按 W 衰减
    decayed_state = old_state * W

    # 当前 token 的 K、V 写入 state
    new_info = write(K, V)

    # 得到新的 state
    state = decayed_state + new_info

    # 当前 token 通过 R 从 state 中读取信息
    output = R * read(state)
```

这段代码不是 RWKV 的真实源码，而是为了帮助理解。真正的 RWKV 实现会更复杂，但整体思想就是：旧信息衰减，新信息写入，state 更新，当前 token 再从 state 中读取信息。

---

## 八、结合例子理解 RWKV 的递推过程

假设输入句子是：

```text
我 喜欢 打 篮球
```

当模型读到“我”时，state 一开始基本为空。模型会根据“我”生成对应的 R、W、K、V，然后把“我”的相关信息写入 state。

读到“喜欢”时，旧 state 中已经有“我”的信息。模型会先让旧 state 按 W 衰减，再把“喜欢”的 K、V 信息写入 state，得到新的 state。此时 state 中已经包含“我”和“喜欢”的信息。

读到“打”时，模型继续对旧 state 做衰减，并写入“打”的信息。此时 state 中开始包含动作相关的信息。

读到“篮球”时，模型会使用已经更新过多次的 state。这个 state 中已经融合了前面的“我”“喜欢”“打”等信息。当前 token “篮球”通过 R 从 state 中读取有用的历史信息，再生成融合上下文后的表示。

这个过程说明，RWKV 不是只看当前 token，而是通过 state 间接利用了前文。它也不是每次都保存完整历史，而是将历史不断压缩和更新到 state 中。

---

## 九、Channel Mix 模块学习

除了 Time Mix，RWKV Block 中还有 Channel Mix。Channel Mix 主要负责处理特征维度的信息。

如果一个 token 向量有 768 维，那么这 768 个维度可以理解为不同的特征通道。Time Mix 主要处理不同 token 之间的关系，而 Channel Mix 主要处理同一个 token 内部不同特征维度之间的关系。

可以简单理解为：

```text
Time Mix：负责看历史，处理 token 之间的时间关系
Channel Mix：负责加工当前 token，处理特征维度之间的关系
```

Channel Mix 的作用有点像 Transformer 中的前馈神经网络。它通过线性变换和非线性激活，对当前 token 的表示进行进一步处理，使模型具有更强的表达能力。

一个 RWKV Block 可以概括为：

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

其中，残差连接可以帮助模型更稳定地训练，也能避免信息在多层传递时丢失太多。

---

## 十、Transformer 相关内容回顾

为了更好理解 RWKV，本周也继续复习了 Transformer 中 Q、K、V 和 KV Cache 的作用。

Transformer 中，每个 token 会通过线性层生成 Q、K、V。Q 表示查询向量，K 表示键向量，V 表示值向量。自注意力机制会让当前 token 的 Q 和所有 token 的 K 计算相关性，再根据注意力权重对 V 进行加权求和。

可以简单理解为：

```text
Q：当前 token 想查什么
K：每个 token 提供什么匹配标签
V：每个 token 真正提供什么内容
```

Transformer 中长期保存的是 `Wq、Wk、Wv` 等模型权重。Q、K、V 是当前输入经过线性层临时生成的中间结果，并不会永久保存到模型文件里。

但是在推理生成时，为了避免每次都重复计算历史 token 的 K、V，Transformer 会临时缓存历史 K、V，也就是 KV Cache。KV Cache 不是模型参数，而是推理过程中保存的缓存。生成越长，KV Cache 越大。

这和 RWKV 的 state 不一样。RWKV 不保存所有历史 token 的 K、V，而是把历史信息递推地压缩进 state 中。

---

## 十一、Mamba 相关内容回顾

本周也继续对比了 Mamba 和 RWKV。Mamba 同样不是通过完整注意力矩阵来保存历史信息，而是通过 state 保存历史。

Mamba 的核心是选择性状态空间模型。普通状态空间模型会用隐藏状态保存历史信息，但更新方式比较固定。Mamba 的改进是让模型根据当前 token 动态决定怎么更新 state、怎么写入信息、怎么读取信息。

可以简单理解为：

```text
当前 token 进入模型
↓
模型判断哪些信息该记住，哪些该弱化
↓
更新 state
↓
从 state 中读取信息
↓
得到当前输出
```

Mamba 和 RWKV 的共同点是，它们都不依赖完整注意力矩阵，也都用 state 保存历史信息。不同点在于，Mamba 依靠选择性状态空间模型来控制 state 更新，而 RWKV 依靠 Time Mix 和 WKV state 递推机制来融合历史信息。

---

## 十二、RWKV、Transformer 和 Mamba 的对比

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
|需要重点理解|Q、K、V、Attention、KV Cache|state、选择性更新|Time Mix、Channel Mix、R/W/K/V、WKV state|

从整体上看，Transformer 更偏向显式关注历史，Mamba 和 RWKV 更偏向把历史压缩进状态。Transformer 的优势是上下文建模能力强，结构成熟；Mamba 的优势是通过选择性状态空间模型提高长序列效率；RWKV 的优势是用 RNN 式 state 推理，同时保留部分 Transformer 式训练优势。

更直观地说：

```text
Transformer：每一步都能回头看历史 token
Mamba：把历史存在 state 里，当前 token 决定怎么更新和读取
RWKV：把历史递推进 state 里，用 Time Mix 和 WKV 融合历史
```

---

## 十三、本周学习收获

通过本周学习，我对 RWKV 的整体结构和核心机制有了更清楚的认识。RWKV 不是普通 Transformer，也不是传统 RNN，而是一种结合了 RNN 状态递推思想和 Transformer 中 K、V、门控、通道混合等思想的序列模型。

本周重点理解了 Time Mix 中 R、W、K、V 的具体作用。R 控制当前 token 从 state 中读取多少历史信息，W 控制旧信息如何衰减，K 决定当前信息如何写入 state，V 表示当前 token 真正提供的信息内容。WKV state 则负责保存递推得到的历史记忆。

另一个重要收获是进一步分清了模型参数、中间结果和推理缓存之间的区别。Transformer 中，`Wq、Wk、Wv` 是模型权重，Q、K、V 是中间结果，KV Cache 是推理缓存。RWKV 中，Time Mix 和 Channel Mix 相关权重是模型参数，而 state 是推理过程中维护的历史记忆。Mamba 中，生成动态参数的权重是模型参数，动态生成的中间量和 state 则参与当前计算。

通过把 Transformer、Mamba 和 RWKV 放在一起比较，可以发现它们都在解决同一个问题：如何让当前 token 更好地利用历史上下文。只是 Transformer 选择显式注意力，Mamba 选择状态空间模型，RWKV 选择 Time Mix 和 WKV state 递推机制。

---

## 十四、后续学习安排


在完成 RWKV、Transformer 和 Mamba 的阶段性学习后，后续计划开始学习 GAN 相关知识。GAN 即生成对抗网络，主要由生成器和判别器组成，通过二者之间的对抗训练，使生成器逐渐学会生成更接近真实数据的样本。
