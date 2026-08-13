---
title: "Transformer 代码实现与 Mamba 选择性状态空间模型"
slug: deep-learning-report-05
publishDate: 2026-06-10
description: "结合 PyTorch 拆解 Transformer 代码实现，并学习 Mamba 的状态空间模型和选择性机制。"
---
## 一、学习概述

本周主要围绕 Transformer 模型的代码实现和 Mamba 模型的基本结构展开学习。Transformer 部分重点结合 PyTorch 代码，从底层的注意力计算开始，依次学习了 `SelfAttentionn`、`MultiHeadAttention`、`FeedForward`、`EncoderLayer`、`DecoderLayer`、`PositionalEncoding`、`Encoder`、`Decoder` 和完整 `Transformer` 类的实现过程，并对各模块中的输入输出维度变化进行了整理。通过对代码的逐层分析，进一步理解了 Transformer 中 Q、K、V 的生成方式、多头注意力的拆分与拼接过程、Encoder 与 Decoder 的连接方式，以及 Decoder 中 Causal Mask 防止看到未来 token 的具体实现。

在 Mamba 模型学习方面，本周结合 IBM Think 网站中关于 Mamba 模型的结构图和公式，对状态空间模型（SSM）、选择性状态空间模型（S6）和 Mamba 模块的基本结构进行了学习。重点理解了 Mamba 与 Transformer 在序列建模方式上的区别：Transformer 通过注意力机制显式计算 token 之间的相关性，而 Mamba 通过状态空间模型将历史信息压缩到隐藏状态中，并通过选择性机制动态控制信息的写入、保留和输出。

## 二、具体学习内容

### （1）Transformer 代码整体结构

本周首先对 Transformer 的整体代码结构进行了梳理。当前代码采用自底向上的实现方式，先定义最基础的注意力计算模块，再逐步组合成完整的 Encoder-Decoder 结构。代码整体结构如下：

```python
SelfAttentionn
 ↓
MultiHeadAttention
 ↓
FeedForward
 ↓
EncoderLayer / DecoderLayer
 ↓
PositionalEncoding
 ↓
Encoder / Decoder
 ↓
Transformer
 ↓
generate_mask 测试
```

其中，`SelfAttentionn` 负责完成最核心的注意力公式计算；`MultiHeadAttention` 在此基础上完成 Q、K、V 的线性映射、多头拆分、注意力计算和多头拼接；`FeedForward` 用于对每个 token 的特征进行非线性变换；`EncoderLayer` 和 `DecoderLayer` 则分别组合注意力模块和前馈网络，构成 Transformer 的基本层结构。最后，`Encoder` 和 `Decoder` 通过多层堆叠形成完整的编码器和解码器，`Transformer` 类负责将二者连接起来，实现从源序列输入到目标序列预测的完整流程。

在测试代码中，源语言词表和目标语言词表大小均设置为 10000，输入源序列 `src` 的形状为 `[32, 10]`，目标序列 `tgt` 的形状为 `[32, 20]`。其中 32 表示 batch size，10 表示源序列长度，20 表示目标序列长度。模型最终输出形状为 `[32, 20, 10000]`，表示每个目标序列位置都会输出一个 10000 维的词表预测分数。
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-01.png)

### （2）SelfAttentionn 模块实现
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-02.png)

`SelfAttentionn` 是当前代码中最底层的注意力计算模块，主要实现缩放点积注意力。其核心公式为：
$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

在该模块中，输入 Q、K、V 的形状分别为：

```python
Q: [batch, heads, seq_len_q, d_k]
K: [batch, heads, seq_len_k, d_k]
V: [batch, heads, seq_len_v, d_v]
```

代码中首先通过：

```python
d_k = Q.size(-1)
```

获取 Q 的最后一维大小，也就是每个 head 中 query 向量的维度。随后使用：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

计算注意力分数。这里 `K.transpose(-2, -1)` 表示交换 K 的最后两个维度，使 Q 和 K 可以进行矩阵乘法。以 Decoder 自注意力为例，Q 和 K 的形状分别为 `[32, 8, 20, 64]` 和 `[32, 8, 20, 64]`，K 转置后变为 `[32, 8, 64, 20]`，因此计算得到的 `scores` 形状为 `[32, 8, 20, 20]`。

接着，如果传入了 mask，则使用：

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

将不可见位置设置为负无穷。这样经过 softmax 后，这些位置的注意力权重会变成 0，从而实现屏蔽效果。最后，代码通过：

```python
attn = self.softmax(scores)
out = torch.matmul(attn, V)
```

得到注意力权重和加权后的输出结果。该模块最终返回 `out` 和 `attn`，其中 `out` 是注意力加权后的结果，`attn` 是注意力权重矩阵，便于后续观察模型关注了哪些位置。


### （3）MultiHeadAttention 模块实现
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-03.png)![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-04.png)
在 `MultiHeadAttention` 中，代码首先定义了 Q、K、V 的三个线性映射层：

```python
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
```

虽然线性层输入和输出维度都是 `d_model`，但三组参数不同，因此可以将同一个输入映射成不同的 Q、K、V 表示。以当前参数为例，`d_model=512`，`n_heads=8`，因此每个注意力头的维度为：
$$
d_k = \frac{d_{\mathrm{model}}}{n_{\mathrm{heads}}}
= \frac{512}{8}
= 64
$$
在 forward 过程中，代码通过：

```python
Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

完成 Q、K、V 的生成与多头拆分。以目标序列输入 `[32, 20, 512]` 为例，经过线性层后形状仍为 `[32, 20, 512]`，然后通过 `view` 变为 `[32, 20, 8, 64]`，再通过 `transpose(1, 2)` 变为 `[32, 8, 20, 64]`。这样做的目的是把 512 维向量拆成 8 个 64 维的小向量，让不同注意力头从不同子空间学习序列特征。

完成注意力计算后，输出形状为 `[32, 8, 20, 64]`。代码再通过：

```python
out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
```

将多个 head 重新拼接。具体变化过程为：

```python
[32, 8, 20, 64] → [32, 20, 8, 64] → [32, 20, 512]
```

随后经过 `self.fc(out)`，将多头拼接后的结果再次线性映射，使不同 head 的信息进一步融合。最后代码通过：

```python
return self.norm(out + q), attn
```

完成残差连接和 LayerNorm。残差连接可以保留原始输入信息，LayerNorm 则可以使每层输出分布更加稳定，有利于深层模型训练。

### （4）FeedForward 前馈网络实现
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-05.png)

`FeedForward` 模块由两层线性层组成：

```python
self.fc1 = nn.Linear(d_model, d_ff)
self.fc2 = nn.Linear(d_ff, d_model)
```

当前代码中 `d_model=512`，`d_ff=2048`，因此该模块的维度变化为：

```python
[batch, seq_len, 512] → [batch, seq_len, 2048] → [batch, seq_len, 512]
```

第一层线性层将特征维度从 512 扩展到 2048，相当于把 token 表示映射到更高维空间中学习更丰富的特征；中间经过 ReLU 激活函数引入非线性，再经过 Dropout 防止过拟合；第二层线性层将维度重新压回 512，保证能够与输入做残差连接。

代码中 forward 过程为：

```python
out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
return self.norm(out + x)
```

可以看出，前馈网络不会改变序列长度，也不会让不同 token 之间直接交互，而是对每个 token 的 512 维特征单独进行非线性变换。token 之间的信息交互主要由注意力机制完成，而 FeedForward 负责增强每个位置自身的表达能力。

### （5）EncoderLayer 与 Encoder 实现
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-06.png)

`EncoderLayer` 由多头自注意力和前馈网络组成。代码中：

```python
out, _ = self.self_attn(src, src, src, src_mask)
out = self.ffn(out)
```

说明 Encoder 中 Q、K、V 都来自同一个 `src`，因此这是自注意力机制。自注意力的作用是让源序列中的每个 token 都能够关注同一句子中的其他 token，从而获得上下文信息。

以当前测试输入为例，源序列 `src` 的初始形状为 `[32, 10]`，经过 `nn.Embedding` 后变成 `[32, 10, 512]`，再经过位置编码后形状仍为 `[32, 10, 512]`。进入 EncoderLayer 后，Q、K、V 经过多头拆分变为 `[32, 8, 10, 64]`，注意力分数矩阵形状为 `[32, 8, 10, 10]`，表示源序列中 10 个 token 两两之间的相关性。

完整 `Encoder` 类中使用 `nn.ModuleList` 堆叠多个 EncoderLayer：
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-07.png)

```python
self.layers = nn.ModuleList([
    EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
])
```

默认情况下，编码器层数为 6 层。因此源序列会连续经过 6 层 EncoderLayer，每一层都会更新 token 的上下文表示，但整体形状始终保持 `[32, 10, 512]`。最后 Encoder 输出 `memory`，其形状为 `[32, 10, 512]`，表示源语言序列的上下文编码结果。

### （6）DecoderLayer 与 Decoder 实现
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-08.png)

`DecoderLayer` 是本周学习中的重点之一。与 EncoderLayer 不同，DecoderLayer 包含三个部分：Masked Self-Attention、Cross-Attention 和 FeedForward。

第一部分是目标序列内部的 Masked Self-Attention：

```python
out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
```

这里 Q、K、V 都来自目标序列 `tgt`，但是会传入 `tgt_mask`，用于屏蔽未来 token。以当前目标序列长度 20 为例，Q、K、V 拆分 head 后形状均为 `[32, 8, 20, 64]`，注意力分数矩阵为 `[32, 8, 20, 20]`。通过 Causal Mask 后，每个位置只能看到当前位置及其之前的位置，不能看到后面的 token。

第二部分是 Cross-Attention：

```python
out, _ = self.cross_attn(out, memory, memory, memory_mask)
```

这里需要特别注意：Encoder 的输出结果只有一个 `memory`，但是这个 `memory` 在 Decoder 中会同时作为 K 和 V 的输入，而 Decoder 当前输出 `out` 作为 Q。因此 Cross-Attention 的输入关系可以表示为：
$$
\begin{aligned}
Q &= \mathrm{DecoderOutput} \\
K &= \mathrm{memory} \\
V &= \mathrm{memory}
\end{aligned}
$$

这也解释了为什么 Transformer 结构图中经常会从 Encoder 到 Decoder 画两条线：并不是 Encoder 输出了两个不同结果，而是同一个 `memory` 同时被送入 Decoder 的 K 和 V 分支。在代码中虽然传入的是同一个 `memory`，但进入 `MultiHeadAttention` 后会分别经过 `W_k` 和 `W_v` 两个不同的线性层，因此最终得到的 K 和 V 表示并不完全相同。

以当前测试维度为例，`memory` 的形状为 `[32, 10, 512]`，Decoder 当前输出 `out` 的形状为 `[32, 20, 512]`。进入 Cross-Attention 后，Q 的形状为 `[32, 8, 20, 64]`，K 和 V 的形状为 `[32, 8, 10, 64]`。注意力分数矩阵形状为 `[32, 8, 20, 10]`，表示目标序列中的 20 个位置分别去源序列的 10 个位置中寻找相关信息。

![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-09.png)
完整 `Decoder` 类同样使用 `nn.ModuleList` 堆叠多个 DecoderLayer。每一层 DecoderLayer 都会先在目标序列内部建模，再通过 Cross-Attention 读取 Encoder 的 `memory`，最后通过 FeedForward 更新每个位置的表示。Decoder 最后一层输出形状为 `[32, 20, 512]`，随后通过：

```python
self.fc_out = nn.Linear(d_model, vocab_size)
```

映射到目标词表大小，最终输出 `[32, 20, 10000]`。

### （7）PositionalEncoding 与 generate_mask 实现
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-10.png)

在位置编码部分，代码首先创建形状为 `[max_len, d_model]` 的位置编码矩阵：

```python
pe = torch.zeros(max_len, d_model)
```

然后使用 `torch.arange` 生成位置索引，并通过 sin 和 cos 函数分别填充偶数维和奇数维：

```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

最后通过：

```python
self.register_buffer('pe', pe)
```

将位置编码注册为 buffer。这样位置编码会随着模型一起保存和移动到 GPU，但不会作为可训练参数参与梯度更新。在 forward 中，代码根据当前输入序列长度截取对应位置编码，并与 embedding 相加，使模型能够获得 token 的顺序信息。


![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-11.png)
在 mask 构造部分，代码定义了 `generate_mask(size)` 函数，用于生成 Decoder 中的 Causal Mask：

```python
mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
return mask == 0
```

当 `size=20` 时，该函数生成形状为 `[20, 20]` 的下三角可见矩阵。矩阵中 True 表示当前位置可见，False 表示未来位置需要被屏蔽。在 Decoder 的注意力计算中，该 mask 可以广播到 `[32, 8, 20, 20]`，从而保证目标序列中每个位置只能关注自己及之前的位置。

### （8）Transformer 整体前向传播与测试结果
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-12.png)

完整 Transformer 类将 Encoder 和 Decoder 连接起来。forward 过程如下：

```python
memory = self.encoder(src, src_mask)
out = self.decoder(tgt, memory, tgt_mask, memory_mask)
return out
```

该流程可以概括为：

```python
src → Encoder → memory
tgt + memory → Decoder → out
```

结合当前测试代码，源序列 `src` 形状为 `[32, 10]`，目标序列 `tgt` 形状为 `[32, 20]`。经过 Encoder 后得到 `memory`，形状为 `[32, 10, 512]`；Decoder 接收目标序列和 `memory`，经过多层 DecoderLayer 后输出 `[32, 20, 512]`；最后经过输出层映射到词表大小，得到最终输出 `[32, 20, 10000]`。

该输出表示：对于 batch 中的 32 个样本，每个目标序列的 20 个位置都会得到一个 10000 维向量，用于表示当前位置预测为目标词表中每个 token 的分数。在训练翻译任务时，可以将该输出与目标标签计算交叉熵损失；在推理生成任务中，则可以取最后一个位置的输出分数，选择概率最高的 token 作为下一个生成词。

### （9）Mamba 模型中的状态空间模型公式

在 Mamba 模型学习方面，本周结合 IBM Think 网站中的结构图和公式，重点理解了状态空间模型（SSM）的基本思想。SSM 的核心是通过隐藏状态保存历史信息，并根据当前输入不断更新状态。

状态方程可以表示为：

$$
h_t = A h_{t-1} + B x_t
$$
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-13.png)

其中，`h_t` 表示当前时刻的隐藏状态，`h_{t-1}` 表示上一时刻的隐藏状态，`x_t` 表示当前输入。矩阵 A 控制历史状态如何保留和演化，矩阵 B 控制当前输入如何写入隐藏状态。

输出方程可以表示为：
$$
y_t = C h_t + D x_t
$$

![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-14.png)

其中，矩阵 C 控制当前隐藏状态如何影响输出，矩阵 D 表示输入对输出的直接影响。在 IBM 网站中的 SSM 图示中，通常会重点展示 A、B、C 三个核心矩阵，而 D 有时会被省略，以突出状态更新和输出生成的主要过程。

通过这两个公式可以看出，Mamba 与 Transformer 的信息存储方式不同。Transformer 会通过注意力机制显式计算当前 token 与其他 token 的关系，而 SSM 会将历史信息压缩到隐藏状态 `h_t` 中。每处理一个新 token，模型都会根据上一时刻状态和当前输入更新隐藏状态，再根据隐藏状态生成输出。

### （10）选择性 SSM 与 Mamba 模块结构

传统 SSM 中，A、B、C 等参数通常对所有输入保持固定，这会限制模型根据当前 token 动态选择信息的能力。Mamba 的核心改进是引入选择性状态空间模型，使部分参数能够随当前输入动态变化。IBM 网站中提到，选择性 SSM 会让步长 `Δ_t` 以及矩阵 `B_t`、`C_t` 成为当前输入 `x_t` 的函数，通常通过线性投影生成：
$$
\begin{aligned}
\Delta_t &= \mathrm{Linear}_{\Delta}(x_t) \\
B_t &= \mathrm{Linear}_{B}(x_t) \\
C_t &= \mathrm{Linear}_{C}(x_t)
\end{aligned}
$$
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-15.png)
其中，`Δ_t` 控制当前输入对隐藏状态更新的影响幅度；`B_t` 控制当前输入如何写入状态；`C_t` 控制当前状态如何影响输出。需要注意的是，`Δ_t`、`B_t`、`C_t` 是根据当前 token 临时生成的中间结果，模型真正长期保存和训练的是生成它们的线性层权重。这一点与 Transformer 中 Q、K、V 的生成方式很相似：Transformer 保存的是 `W_q`、`W_k`、`W_v`，而不是每个 token 产生的 Q、K、V 中间值。

Mamba 模块的基本结构可以概括为以下流程：
![学习报告 5 配图](/assets/blog/deep-learning/report-05/image-16.png)
```python
输入 x
 ↓
线性投影扩展维度
 ↓
分成 x_proj 和 z_proj 两条路径
 ↓
x_proj → 1D Conv → SiLU → 选择性 SSM
z_proj → SiLU 门控路径
 ↓
SSM 输出 y 与 z_act 逐元素相乘
 ↓
线性投影回原维度
 ↓
与原输入做残差连接
 ↓
输出上下文感知后的表示
```

结合 IBM 网站中的 Mamba 模块图可以理解：Mamba 首先把输入向量扩展到更高维度，例如从 512 维扩展到 1024 维，然后分成两条路径。一条路径进入 SSM 分支，先通过一维卷积提取局部相邻 token 的特征，再通过 SiLU 激活函数和选择性 SSM 建模长距离依赖；另一条路径进入门控分支，用于控制 SSM 输出中哪些信息应该被增强，哪些信息应该被抑制。之后，SSM 输出与门控分支输出逐元素相乘，再通过线性层投影回原始维度，并与输入残差相加，得到最终输出。


### （11）Transformer 与 Mamba 的对比理解

通过本周学习，我对 Transformer 和 Mamba 的序列建模方式有了更清晰的认识。Transformer 的核心是注意力机制，它通过 Q、K、V 显式计算 token 之间的相关性。以当前实现代码为例，在 Decoder 的自注意力中，注意力分数矩阵形状为 `[32, 8, 20, 20]`；如果序列长度继续增加，注意力矩阵也会随之变大。因此 Transformer 在建模复杂上下文方面能力较强，但长序列场景下计算和显存开销较大。

Mamba 则基于状态空间模型，通过隐藏状态保存历史信息，并通过选择性机制控制哪些信息被写入、保留和输出。它不需要像 Transformer 那样对所有 token 两两计算注意力，而是通过状态更新来建模序列信息。因此，Mamba 在长序列建模中具有较好的效率优势。

可以简单概括为：

```python
Transformer：通过注意力矩阵显式查看历史 token
Mamba：通过隐藏状态压缩历史信息，并选择性更新
```

因此，两类模型各有优势。Transformer 适合复杂上下文交互和精细 token 对齐，Mamba 更适合长序列、高效率推理和低内存场景。后续如果继续深入学习，可以重点对比 Transformer 中的 KV Cache 和 Mamba 中的隐藏状态 `h_t`，进一步理解两类模型在推理阶段保存历史信息的方式。

## 三、后续学习安排

后续在 Mamba 部分，将继续整理选择性状态空间模型的基本原理，重点理解隐藏状态更新过程以及 `Δ`、`B`、`C` 等参数在模型中的作用。同时，后续计划开始学习 RWKV 模型，初步了解其整体结构、基本思想以及它与 Transformer、Mamba 在序列建模方式上的区别。
