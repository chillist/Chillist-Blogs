---
title: "Transformer 核心机制：从词嵌入到编解码结构"
slug: deep-learning-report-04
publishDate: 2026-06-03
description: "系统学习 Transformer 的词嵌入、自注意力、多头注意力、掩码、归一化与编解码结构。"
---
## 一、学习概述

本阶段开始学习 Transformer 模型的基础结构和核心原理。Transformer 是一种用于处理序列数据的深度学习模型，常见于机器翻译、文本生成、问答系统和大语言模型中。与传统 RNN 按顺序处理文本不同，Transformer 主要依靠注意力机制来建模词与词之间的关系，因此可以更好地进行并行计算。

本阶段主要学习了词向量表示、自注意力机制、多头注意力机制、掩码机制、层归一化、编码器与解码器结构，以及 Transformer 在训练机器翻译任务时的基本流程。

---

## 二、独热向量与词嵌入

### 1. 独热向量

独热向量，英文是 **One-hot Vector**，用来表示某个词在词表中的编号。它的特点是只有一个位置为 1，其余位置全是 0。

例如词表为：

```text
0: 我
1: 喜欢
2: 猫
3: 狗
```

那么“猫”的独热向量可以表示为：

```text
[0, 0, 1, 0]
```

独热向量可以表示“这个词是谁”，但它有两个问题：一是维度很高，二是无法表达词语之间的语义关系。例如“猫”和“狗”在 one-hot 中看起来完全无关，但实际上它们都属于动物。

---

### 2. 词嵌入矩阵

词嵌入，英文是 **Embedding**，作用是把词 ID 转换成连续的低维向量。词嵌入矩阵可以理解为一张可训练的“词向量查询表”。

在 PyTorch 中，常见的词嵌入矩阵形状是：

```text
[V, d]
```

其中：

```text
V：词表大小
d：词向量维度
```

也就是说：

```text
每一行对应一个词
每一列对应一个维度
```

例如：

```text
Embedding Matrix =

我      [0.10, 0.20, 0.30]
喜欢    [0.40, 0.50, 0.60]
猫      [0.70, 0.80, 0.90]
狗      [0.11, 0.22, 0.33]
```

如果输入词是“猫”，它的 ID 是 2，那么就取词嵌入矩阵的第 2 行，得到“猫”的词向量。

需要注意的是，有些 PPT 会把词嵌入矩阵画成 `[d, V]`，也就是转置后的形式。这种画法也可以成立，只是矩阵方向约定不同。实际写 PyTorch 代码时，通常按 `[V, d]` 理解更方便。

![图1 独热向量与词嵌入矩阵示意图](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-02.png)

**图1 独热向量与词嵌入矩阵示意图**

图注：独热向量用于表示词在词表中的位置，词嵌入矩阵则相当于一张可训练的词向量表。通过词 ID 或独热向量可以从词嵌入矩阵中取出对应的连续词向量，供 Transformer 后续计算使用。


---

## 三、Q、K、V 与自注意力机制

### 1. 自注意力机制

自注意力，英文是 **Self-Attention**，作用是让句子中的每个 token 都可以关注句子中其他 token，从而获得上下文信息。

这里的 **Token** 指模型处理文本时的基本单位，可以是一个字、一个词，也可以是子词。

例如句子：

```text
我 喜欢 打 篮球
```

当模型处理“打”这个词时，它可能需要重点关注“篮球”，因为“打篮球”是一个整体语义。自注意力机制就是用来建模这种关系的。

---

### 2. Q、K、V 的含义

在自注意力机制中，每个 token 的输入向量都会生成三个向量：

```text
Q = Query，查询向量
K = Key，键向量
V = Value，值向量
```

它们的作用可以这样理解：

```text
Q：当前 token 想查询什么信息
K：每个 token 提供的匹配标签
V：每个 token 真正提供的信息内容
```

一句话总结：

```text
Q 和 K 决定关注谁，V 提供被关注的信息。
```

公式为：

```text
Q = XWq
K = XWk
V = XWv
```

其中 `X` 是输入矩阵，`Wq、Wk、Wv` 是模型训练得到的权重矩阵。

---

### 3. 自注意力计算过程

自注意力的核心公式是：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

其中：

```text
QKᵀ：计算 token 之间的相关性分数
√d_k：缩放因子，防止分数过大导致 softmax 过于极端
softmax：把相关性分数转换成注意力权重
V：提供真正被加权汇总的信息
```

如果输入序列长度为 `L`，每个 Q/K/V 向量维度为 `d_k`，那么：

```text
Q.shape = L × d_k
K.shape = L × d_k
V.shape = L × d_k
QKᵀ.shape = L × L
A.shape = L × L
AV.shape = L × d_k
```

其中 `A` 是注意力权重矩阵，`AV` 是最终输出。
`AV` 表示每个 token 根据注意力权重，从所有 token 的 V 中融合信息后得到的新表示。

![图2 自注意力机制矩阵计算流程](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-03.png)

**图2 自注意力机制矩阵计算流程**

图注：输入矩阵 X 分别经过三个权重矩阵得到 Q、K、V。Q 和 K 用于计算 token 之间的相关性，经过缩放和 softmax 后得到注意力权重矩阵 A，最后 A 与 V 相乘，得到融合上下文信息后的新 token 表示。


---

## 四、多头注意力机制

多头注意力，英文是 **Multi-Head Attention**，简称 **MHA**。

它的核心思想是：不只计算一次注意力，而是并行计算多组注意力，让模型从不同角度理解词与词之间的关系。

一个 **head** 就是一组独立的注意力计算。每个 head 都有自己的：

```text
Wq、Wk、Wv
```

如果有 8 个 head，每个 head 都会单独生成一组 Q、K、V，并计算一次 attention。

每个 head 的输出形状为：

```text
L × d_k
```

8 个 head 拼接后，形状为：

```text
L × 8d_k
```

这里要注意，拼接是在**特征维度**上拼接，而不是在序列长度维度上拼接。也就是说，token 数量 `L` 不变，只是每个 token 的特征维度变长。

随后会通过输出投影矩阵 `W^O` 映射回模型维度：

```text
Concat(head_1, ..., head_h) W^O
```

如果原始模型维度是 `d`，最终输出仍然是：

```text
L × d
```

这样后续才能接残差连接和下一层 Transformer。

![图3 多头注意力机制结构示意图](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-04.png)

**图3 多头注意力机制结构示意图**

图注：多个注意力头分别使用不同的 Wq、Wk、Wv 生成多组 Q、K、V，并独立完成注意力计算。多个 head 的输出会在特征维度上拼接，再通过输出投影矩阵 W^O 融合回模型维度。


---

## 五、掩码机制

Transformer 中常见两种掩码：

```text
Padding Mask：填充掩码
Causal Mask：因果掩码
```

---

### 1. Padding Mask

Padding，中文叫 **填充**。由于同一个 batch 中不同句子长度可能不一样，为了让 GPU 并行计算，需要把短句子补齐到相同长度。

例如：

```text
我 喜欢 篮球
我 喜欢 打 篮球
```

补齐后可能变成：

```text
我 喜欢 篮球 [PAD]
我 喜欢 打 篮球
```

其中 `[PAD]` 是填充 token，没有实际语义。

Padding Mask 的作用是告诉模型：

```text
不要关注 [PAD] 位置
```

在注意力分数中，会给 padding 位置加上 `-∞`：

```text
Scores = QKᵀ / √d_k + Mask
```

经过 softmax 后，`-∞` 对应的位置会变成 0，因此模型就不会从 padding token 中取信息。

![图4 Padding Mask 填充掩码示意图](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-05.png)

**图4 Padding Mask 填充掩码示意图**

图注：同一个 batch 中的句子需要通过 [PAD] 补齐到相同长度。Padding Mask 会在注意力分数中屏蔽 padding 位置，使其经过 softmax 后权重为 0，从而避免模型关注无效 token。


---

### 2. Causal Mask

Causal Mask，中文叫 **因果掩码**，也叫 **Look-ahead Mask**。

它主要用于 Decoder 的自注意力中，作用是：

```text
防止模型在训练时偷看未来词
```

例如目标句子是：

```text
I like playing ball
```

训练时 Decoder 输入是：

```text
<bos> I like playing ball
```

但模型在预测 `like` 时，只能看到：

```text
<bos> I
```

不能看到后面的：

```text
playing ball
```

所以 Causal Mask 会把未来位置遮住，使注意力矩阵变成下三角形式。这样模型只能根据当前词和前面的词预测下一个词。

![图5 Causal Mask 因果掩码示意图](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-06.png)

**图5 Causal Mask 因果掩码示意图**

图注：Decoder 训练时需要防止模型提前看到未来词。Causal Mask 会把当前位置之后的 token 屏蔽掉，使模型在每个位置只能关注当前词及其之前的词，从而学习根据前文预测下一个词。


---

## 六、层归一化

层归一化，英文是 **Layer Normalization**，简称 **LayerNorm** 或 **LN**。

它的作用是让每个 token 的特征向量分布更稳定，从而帮助模型训练。

公式为：

```text
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
```

其中：

```text
x：当前 token 的特征向量
μ：均值
σ：标准差
ε：极小值，防止除以 0
γ：可学习的缩放参数
β：可学习的偏移参数
```

需要注意，`σ` 是标准差，`σ²` 才是方差。

在 Transformer 中，LayerNorm 通常是对每个 token 的特征维度做归一化。假设输入形状为：

```text
batch_size × seq_len × hidden_dim
```

LayerNorm 一般作用在最后一维：

```text
hidden_dim
```

也就是对每个 token 自己的特征向量进行归一化，而不是跨 batch 统计。
- 层归一化和批量归一化的目标相同，但是层归一化是基于特征维度进行归一化的
- 层归一化和批量归一化的区别在于：批量归一化在 d 的维度上找出一个矩阵，将其均值变成 0 ，方差变成 1，层归一化每次选的是一个元素，也就是每个 batch 里面的一个样本进行归一化
- 尽管批量归一化在计算机视觉中被广泛应用，但是在自然语言处理任务中，批量归一化通常不如层归一化的效果好，因为**在自然语言处理任务中，输入序列的长度通常是变化的**
- 虽然在做层归一化的时候，长度也是变化的，但是至少来说还是在一个**单样本**中，不管批量多少，都给定一个特征，这样对于变化的长度来讲，稍微稳定一点，不会因为长度变化，导致稳定性发生很大的变化
![学习报告 4 配图](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-01.png)

![图6 LayerNorm 层归一化示意图](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-07.png)

**图6 LayerNorm 层归一化示意图**

图注：LayerNorm 通常对每个 token 的特征维度进行归一化，通过减去均值、除以标准差，再使用可学习参数 γ 和 β 进行缩放和平移，从而稳定 Transformer 的训练过程。


---

## 七、Encoder 和 Decoder

### 1. Encoder

它的作用是读取输入序列，并将输入序列转换成上下文表示。

例如输入：

```text
我 喜欢 打 篮球
```

Encoder 会输出每个 token 融合上下文后的表示，作为后续 Decoder 参考的信息。

Encoder 主要由以下结构组成：

```text
Multi-Head Attention
Add & Norm
Feed Forward
Add & Norm
```

---

### 2. Decoder

它的作用是根据 Encoder 输出的编码信息，以及已经生成的目标语言前文，预测下一个词。

Decoder 中包含三个主要部分：

```text
Masked Multi-Head Attention
Cross-Attention
Feed Forward
```

其中：

```text
Masked Multi-Head Attention：目标序列内部做注意力，但不能看未来词
Cross-Attention：Decoder 去关注 Encoder 输出
Feed Forward：进一步变换每个 token 的表示
```

---

## 八、交叉注意力机制

交叉注意力，英文是 **Cross-Attention**。

它出现在 Decoder 中，用来让 Decoder 参考 Encoder 的输出。

自注意力是：

```text
同一个序列内部互相看
```

交叉注意力是：

```text
Decoder 序列去看 Encoder 序列
```

在交叉注意力中：

```text
Q 来自 Decoder
K 和 V 来自 Encoder
```

也就是说：

```text
Q：Decoder 当前需要查询什么信息
K：Encoder 每个 token 提供的匹配标签
V：Encoder 每个 token 真正提供的信息
```

假设 Encoder 序列长度是 `L_E`，Decoder 序列长度是 `L_D`，则：

```text
Q.shape = L_D × d_k
K.shape = L_E × d_k
V.shape = L_E × d_k
```

所以：

```text
QKᵀ.shape = L_D × L_E
A.shape = L_D × L_E
AV.shape = L_D × d_k
```

这表示 Decoder 中每个位置都会去关注 Encoder 中所有位置，从源语言句子中提取有用信息。

例如翻译时，Decoder 在生成 `ball` 时，可能会重点关注 Encoder 中的“篮球”。

![图7 Cross-Attention 交叉注意力矩阵计算过程](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-08.png)

**图7 Cross-Attention 交叉注意力矩阵计算过程**

图注：交叉注意力中，Q 来自 Decoder，K 和 V 来自 Encoder。Decoder 利用自己的 Q 去匹配 Encoder 输出的 K，得到目标序列对源序列的注意力权重，再用该权重对 Encoder 的 V 进行加权求和，得到融合源句子信息后的 Decoder 表示。


---

## 九、Transformer 训练过程

以机器翻译为例：

```text
中文输入：我 喜欢 打 篮球
英文目标：I like playing basketball <eos>
```

训练时，Encoder 输入中文句子：

```text
我 喜欢 打 篮球
```

Decoder 输入目标句子右移一位后的结果：

```text
<bos> I like playing basketball
```

正确标签是：

```text
I like playing basketball <eos>
```

其中：

```text
<bos>：句子开始标记
<eos>：句子结束标记
```

这种把目标句子右移一位作为 Decoder 输入的方式，叫 **shift right**。

训练时对应关系为：

| Decoder 输入 | 正确预测目标  |
| ---------- | ------- |
| `<bos>`    | `I`     |
| `I`        | `like` |
| `like`     | `playing` |
| `playing`  | `basketball` |
| `basketball` | `<eos>` |

模型会在每个位置输出一个词表概率分布，然后和正确答案计算 loss。

**Loss**，中文叫 **损失**，表示模型预测和真实答案之间的差距。训练的目标就是让 loss 逐渐变小。

训练时常用 **Teacher Forcing**，中文可以叫 **教师强制**。意思是训练过程中直接把正确的前文输入给 Decoder，让模型学习根据正确前文预测下一个词。

真正推理时，模型没有标准答案，只能一个词一个词生成：

```text
输入 <bos> → 预测 I
输入 <bos> I → 预测 like
输入 <bos> I like → 预测 playing
输入 <bos> I like playing → 预测 basketball
输入 <bos> I like playing basketball → 预测 <eos>
```

预测到 `<eos>` 后，句子生成结束。

![图8 Transformer 编码器-解码器训练流程](/Chillist-Blogs/assets/blog/deep-learning/report-04/image-09.png)

**图8 Transformer 编码器-解码器训练流程**

图注：Transformer 训练机器翻译任务时，Encoder 负责编码源语言句子，Decoder 接收右移后的目标语言序列，并结合 Encoder 输出预测下一个 token。每个位置的预测结果都会与真实标签计算 loss，用于更新模型参数。


---

## 十、学习重点总结

本阶段主要理解了 Transformer 的以下核心内容：

```text
1. 文本需要先通过 one-hot 或 token id 表示，再经过 embedding 转换成词向量。
2. Self-Attention 通过 Q、K 计算注意力权重，再用权重加权 V 得到新的 token 表示。
3. Multi-Head Attention 通过多个 head 从不同角度学习 token 之间的关系。
4. Padding Mask 用于屏蔽补齐位置，Causal Mask 用于防止 Decoder 偷看未来词。
5. LayerNorm 用于稳定每个 token 的特征分布，提高训练稳定性。
6. Encoder 负责理解输入句子，Decoder 负责根据 Encoder 信息和已生成前文预测下一个词。
7. Cross-Attention 中 Q 来自 Decoder，K 和 V 来自 Encoder，用于实现目标语言对源语言的对齐和参考。
8. Transformer 训练时使用右移后的目标序列作为 Decoder 输入，并通过每个位置的 loss 来更新模型参数。
```

---

## 十一、后续学习安排

后续继续学习 Transformer 的完整代码实现

```text
1. nn.Embedding 如何生成词向量；
2. Q、K、V 的矩阵维度如何在代码中变化；
3. Multi-Head Attention 如何拆分和拼接多个 head；
4. Padding Mask 和 Causal Mask 如何在代码中构造；
5. EncoderLayer 和 DecoderLayer 的具体实现；
```

再继续学习 Mamba 和 RWKV
