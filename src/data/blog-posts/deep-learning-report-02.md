---
title: "从数据加载到迁移学习：ResNet18 猫狗分类实践"
slug: deep-learning-report-02
publishDate: 2026-05-19
description: "围绕猫狗分类项目，梳理 PyTorch Dataset、DataLoader、数据划分与 ResNet18 迁移学习流程。"
---
## 一、学习概述

本阶段在前期完成 CIFAR-10 图像分类实验，并学习 AlexNet、VGG、ResNet18、DenseNet 等经典卷积神经网络结构的基础上，进一步进入更贴近真实任务的图像分类项目。本次学习以 **猫狗识别分类** 为实践内容，使用 KaggleHub 下载的 `cat-dog-images-for-classification` 数据集，围绕真实图片数据的读取、数据集划分、模型迁移学习、训练与测试流程等内容展开。

相比此前使用 PyTorch 内置数据集或封装较完善的 CIFAR-10 数据集，本阶段的数据处理过程更加完整。猫狗数据集由 **图片文件夹** 和 **CSV 标签文件** 共同组成，因此需要自行编写数据读取逻辑，构建自定义 `Dataset`，再进一步生成可供模型训练的 `DataLoader`。通过这一过程，进一步理解了 PyTorch 中：

- `Dataset` 与 `DataLoader` 的职责区别；
    
- CSV 标签与图片路径之间的对应关系；
    
- 训练集、验证集、测试集的划分方式；
    
- 训练集数据增强与验证、测试集固定预处理之间的区别。
    

模型方面，本阶段采用 **ResNet18 迁移学习** 完成猫狗二分类任务。通过加载 ImageNet 预训练权重，冻结主干网络参数，仅训练最后的分类层，实现从 1000 类 ImageNet 分类任务到猫狗 2 分类任务的迁移。学习过程中，重点理解了：

- 预训练模型为什么能迁移到新任务；
    
- `freeze_backbone=True` 的意义；
    
- 为什么需要替换原本的 `fc` 分类层；
    
- 最终网络输出的两个值分别如何对应“猫”和“狗”。
    

训练方面，继续完善了完整训练流程，并新增 `tqdm` 训练进度条。对 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`、`model.train()`、`model.eval()`、`@torch.no_grad()` 等核心代码的作用进行了进一步梳理。经过本阶段学习，对一个真实图像分类项目从数据准备到模型预测的完整流程有了更系统的认识。

---

## 二、具体学习内容

### （1）猫狗分类项目与数据集准备

本阶段使用 KaggleHub 下载猫狗分类数据集：

```python
import kagglehub

path = kagglehub.dataset_download(
    "ashfakyeafi/cat-dog-images-for-classification"
)
```

该数据集主要由两部分组成：

- `cat_dog.csv`：记录图片文件名与类别标签；
    
- `cat_dog/`：保存所有猫狗图片文件。
    

CSV 文件中主要包含：

|字段|含义|
|---|---|
|`image`|图片文件名|
|`labels`|图片类别标签|

其中标签含义约定为：

```python
CLASS_NAMES = {
    0: "cat",
    1: "dog"
}
```

也就是说：

```text
0 表示猫
1 表示狗
```


数据集信息基本配置
```python
# =========================================================  
# 1. 基本配置  
# =========================================================  
  
DATASET_ID = "ashfakyeafi/cat-dog-images-for-classification"  
  
# 当前项目根目录  
PROJECT_ROOT = Path(__file__).resolve().parent  
  
# 当前项目中的 dataset 文件夹  
DATASET_DIR = PROJECT_ROOT / "dataset"  
  
# 猫狗数据集专属目录  
LOCAL_DATASET_DIR = DATASET_DIR / "cat_dog_images_for_classification"  
  
# 类别映射  
CLASS_NAMES = {  
    0: "cat",  
    1: "dog"  
}

```

在数据集定位函数 `get_dataset_root()` 中，程序会：

1. 先检查本地项目目录下是否已有完整数据；
    
2. 如果有，则直接复用；
    
3. 如果没有，再自动下载；
    
4. 如果目录存在但数据不完整，则主动报错，避免重复下载和目录冲突。
    


**猫狗分类项目数据目录结构**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-01.png)

---

### （2）训练集与验证集图像预处理

由于真实图片的原始尺寸可能各不相同，模型训练前必须先统一尺寸，并进行必要的数据增强和归一化处理。

训练集使用：

```python
def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
```

其处理流程为：

```text
Resize
→ RandomResizedCrop
→ RandomHorizontalFlip
→ ToTensor
→ Normalize
```

各步骤作用如下：

- `Resize((256, 256))`：先将图片统一放大到固定尺寸；
    
- `RandomResizedCrop(224)`：随机裁剪并调整为 224×224；
    
- `RandomHorizontalFlip(p=0.5)`：以 50% 概率进行水平翻转；
    
- `ToTensor()`：将 PIL 图片转换为 PyTorch Tensor；
    
- `Normalize(...)`：按 ImageNet 预训练模型常用均值与标准差进行归一化。
    

训练集进行随机裁剪与翻转，是为了增加样本多样性，降低模型对固定构图的依赖，提高泛化能力。

验证集与测试集使用固定预处理，不再进行随机增强：

```text
Resize
→ ToTensor
→ Normalize
```

这样可以保证每次验证和测试时输入保持稳定，评估结果更具可比性。

**训练集图像预处理代码**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-02.png)


**验证集与测试集图像预处理代码**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-03.png)


---

### （3）自定义 CatDogDataset 数据集类

由于猫狗数据的标签存储在 CSV 文件中，不能直接使用 PyTorch 内置数据集接口，因此需要自定义数据集类：

```python
class CatDogDataset(Dataset):
```

该类继承自 PyTorch 的 `Dataset` 基类。它的作用是：

> 根据 CSV 中记录的图片文件名和标签，读取对应图片，并返回模型训练所需要的 `(image, label)`。

完整流程如下：

```text
index
→ 查找 DataFrame 对应行
→ 获取 image 文件名
→ 获取 labels 标签
→ 拼接图片完整路径
→ 打开图片
→ 执行 transform
→ 返回 image, label
```

核心代码包括：

```python
def __len__(self):
    return len(self.dataframe)
```

用于返回当前数据集包含多少个样本。

```python
def __getitem__(self, index):
    row = self.dataframe.iloc[index]
```

用于根据索引取出某一行样本信息。

```python
image_name = row["image"]
label = int(row["labels"])
```

用于获得图片名和数字标签。

```python
image_path = self.image_dir / image_name
```

用于拼接图片完整路径。

```python
image = Image.open(image_path).convert("RGB")
```

用于读取图片并统一转换为 RGB 三通道格式。

最后：

```python
return image, label
```

将处理好的图片与标签返回。


**CatDogDataset 自定义数据集类**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-04.png)


---

### （4）关于 Dataset 的理解与重点疑问

#### 1. `Dataset` 是什么？

在：

```python
class CatDogDataset(Dataset):
```

中，`Dataset` 是 PyTorch 提供的数据集基类。它规定了一个自定义数据集通常应具备：

```python
__len__()
__getitem__()
```

两个方法。

`Dataset` 负责：

```text
如何获取单个样本
```

而不是负责按批次训练。

---

#### 2. 为什么 `dataframe: pd.DataFrame` 要传进类中？

```python
dataframe: pd.DataFrame
```

表示传入的是一个 Pandas 表格对象。  
这个表格负责记录：

- 哪张图片属于哪个样本；
    
- 对应标签是猫还是狗。
    

例如：

|image|labels|
|---|--:|
|cat.001.jpg|0|
|dog.002.jpg|1|

后续 `__getitem__()` 就是通过这个表格找到图片名与标签。

---

#### 3. 划分数据集时已经打乱了，索引还能正确找到图片吗？

可以。

`train_test_split()` 打乱的是 **表格行顺序**，而不是打乱每一行内部的图片名和标签对应关系。比如原来：

|image|labels|
|---|--:|
|cat.001.jpg|0|
|dog.002.jpg|1|

即使划分后顺序变了，单行内部仍然保持：

```text
dog.002.jpg 仍然对应标签 1
cat.001.jpg 仍然对应标签 0
```

之后再通过：

```python
reset_index(drop=True)
```

只是重新把行号整理成：

```text
0, 1, 2, 3...
```

并不会改变图片和标签的绑定关系。

---

### （5）训练集、验证集、测试集划分

在 `get_catdog_loaders()` 中，首先读取 CSV：

```python
df = pd.read_csv(csv_path)
```

然后通过两次 `train_test_split()` 完成：

```text
训练集 80%
验证集 10%
测试集 10%
```

第一次划分：

```python
train_df, temp_df = train_test_split(
    df,
    test_size=val_ratio + test_ratio,
    stratify=df["labels"],
    random_state=random_state
)
```

假设：

```python
val_ratio = 0.1
test_ratio = 0.1
```

那么：

```text
80% → 训练集
20% → 临时集
```

第二次划分：

```python
relative_test_ratio = test_ratio / (val_ratio + test_ratio)
```

得到：

```text
0.1 / 0.2 = 0.5
```

表示：

> 在临时集内部，一半作为验证集，一半作为测试集。

最终实现：

```text
80% train
10% val
10% test
```

代码中还设置了：

```python
stratify=df["labels"]
```

其作用是让猫和狗在训练集、验证集、测试集中的比例尽量接近，避免某个集合中类别比例严重失衡。


**图6 训练集、验证集与测试集划分代码**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-05.png)


---

### （6）Dataset 到 DataLoader 的创建过程

本阶段对 `DataLoader` 的理解进一步加深。完整链路如下：

```text
CSV + 图片目录
→ DataFrame
→ train_df / val_df / test_df
→ CatDogDataset
→ DataLoader
```

先创建 Dataset：

```python
train_dataset = CatDogDataset(
    dataframe=train_df,
    image_dir=image_dir,
    transform=get_train_transform(image_size)
)
```

```python
val_dataset = CatDogDataset(
    dataframe=val_df,
    image_dir=image_dir,
    transform=get_eval_transform(image_size)
)
```

```python
test_dataset = CatDogDataset(
    dataframe=test_df,
    image_dir=image_dir,
    transform=get_eval_transform(image_size)
)
```

然后再创建 DataLoader：

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
```

DataLoader 的主要作用是：

- 按 `batch_size` 将多个单样本组成一批；
    
- 训练时根据 `shuffle=True` 打乱数据访问顺序；
    
- 支持多进程读取；
    
- 在 GPU 训练时通过 `pin_memory` 优化数据传输。
    

训练集设置：

```python
shuffle=True
```

是为了每轮训练时让数据顺序不同，提高训练稳定性。

验证集和测试集设置：

```python
shuffle=False
```

是为了保证评估流程稳定一致。

最终取出一批数据：

```python
images, labels = next(iter(train_loader))
```

若：

```python
batch_size = 32
image_size = 224
```

则输出形状通常为：

```text
images.shape = [32, 3, 224, 224]
labels.shape = [32]
```


**DataLoader 构建代码**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-06.png)
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-07.png)


---

### （7）迁移学习版 ResNet18 模型

本阶段使用的是：

```python
CatDogResNet18
```

模型结构。

核心思路是：

> 加载已经在 ImageNet 上训练好的 ResNet18，保留其强大的图像特征提取能力，并将最后分类层改造成猫狗二分类。

模型代码：

```python
weights = ResNet18_Weights.DEFAULT
self.model = resnet18(weights=weights)
```

表示加载预训练 ResNet18。

如果：

```python
freeze_backbone=True
```

则执行：

```python
for param in self.model.parameters():
    param.requires_grad = False
```

这表示冻结主干网络参数，使其不参与训练更新。

原始 ResNet18 最后一层用于 ImageNet 1000 分类：

```text
Linear(512 → 1000)
```

当前猫狗任务只需要两类，因此替换为：

```python
self.model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes)
)
```

即：

```text
512 维特征
→ Dropout
→ Linear(512 → 2)
```

最后两个输出分别对应：

```text
第 0 类：cat
第 1 类：dog
```


**迁移学习版 ResNet18 模型代码**
![学习报告 2 配图](/assets/blog/deep-learning/report-02/image-08.png)

---

### （8）关于模型如何知道猫和狗的疑问

训练结束后，模型会输出两个值，例如：

```text
[2.4, 0.7]
```

这两个值不是直接写着“猫”和“狗”，而是对应两个类别位置：

```text
第 0 位 → cat
第 1 位 → dog
```

这个对应关系来自训练标签：

```python
CLASS_NAMES = {
    0: "cat",
    1: "dog"
}
```

训练时，如果输入是一张猫图，真实标签为：

```text
0
```

损失函数就会推动模型：

```text
让第 0 个输出值更高
让第 1 个输出值更低
```

如果输入是一张狗图，真实标签为：

```text
1
```

训练就会推动模型：

```text
让第 1 个输出值更高
让第 0 个输出值更低
```

因此，模型并不是“天生知道”哪个值代表猫狗，而是在训练过程中，被标签规则逐步教会：

```text
猫图 → 输出第 0 位更高
狗图 → 输出第 1 位更高
```

预测时：

```python
_, predicted = torch.max(outputs, dim=1)
```

会找到分数最大的类别索引，再映射回：

```python
CLASS_NAMES[predicted_label]
```

得到最终文字结果。

---

### （9）训练函数 train_one_epoch

训练函数的主要作用是：

> 完成一个 epoch 内所有 batch 的训练，并返回该轮平均损失和平均准确率。

训练流程如下：

```text
进入训练模式
→ 遍历每个 batch
→ 将数据送入 GPU
→ 梯度清零
→ 前向传播
→ 计算 loss
→ 反向传播
→ 参数更新
→ 统计 loss 和 acc
→ 更新进度条
```

关键代码：

```python
model.train()
```

表示切换为训练模式。

```python
optimizer.zero_grad()
```

表示每个 batch 开始前清空上一个 batch 的梯度。

这是因为 PyTorch 默认会累加梯度，如果不清零，那么当前 batch 的梯度会和上一批梯度叠加，从而影响当前参数更新。

训练核心三步：

```python
loss.backward()
optimizer.step()
```

表示：

- `loss.backward()`：根据损失计算梯度；
    
- `optimizer.step()`：根据梯度更新模型参数。
    

同时使用：

```python
progress_bar = tqdm(loader, desc="Training", leave=False)
```

给训练过程加入实时进度条。


**单轮训练函数代码**
```python
def train_one_epoch(model, loader, criterion, optimizer, device):  
    model.train()  
  
    total_loss = 0.0  
    correct = 0  
    total = 0  
  
    progress_bar = tqdm(loader, desc="Training", leave=True)  
  
    for images, labels in progress_bar:  
        images = images.to(device)  
        labels = labels.to(device)  
  
        optimizer.zero_grad()  
  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
  
        loss.backward()  
        optimizer.step()  
  
        total_loss += loss.item() * images.size(0)  
  
        _, predicted = torch.max(outputs, dim=1)  
        correct += (predicted == labels).sum().item()  
        total += labels.size(0)  
  
        current_acc = correct / total  
  
        progress_bar.set_postfix(  
            loss=f"{loss.item():.4f}",  
            acc=f"{current_acc:.4f}"  
        )  
  
    avg_loss = total_loss / total  
    avg_acc = correct / total  
  
    return avg_loss, avg_acc
```

---

### （10）验证函数 evaluate

验证与测试阶段不需要更新模型参数，因此函数使用：

```python
@torch.no_grad()
```

其作用是：

> 不记录梯度，不构建反向传播计算图，从而节省显存并提高评估速度。

同时：

```python
model.eval()
```

表示将模型切换到评估模式。

这对：

- Dropout；
    
- BatchNorm；
    

非常重要。

训练模式下：

- Dropout 会随机失活部分神经元；
    
- BatchNorm 会使用当前 batch 统计量。
    

评估模式下：

- Dropout 关闭；
    
- BatchNorm 使用训练过程中累计得到的统计信息。
    

验证函数整体流程为：

```text
进入评估模式
→ 遍历验证集或测试集
→ 前向传播
→ 计算 loss
→ 统计预测正确数量
→ 计算平均 loss 与平均 acc
```


**验证与测试函数代码**
```python
@torch.no_grad()  
def evaluate(model, loader, criterion, device):  
    model.eval()  
  
    total_loss = 0.0  
    correct = 0  
    total = 0  
  
    for images, labels in loader:  
        images = images.to(device)  
        labels = labels.to(device)  
  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
  
        total_loss += loss.item() * images.size(0)  
  
        _, predicted = torch.max(outputs, dim=1)  
        correct += (predicted == labels).sum().item()  
        total += labels.size(0)  
  
    avg_loss = total_loss / total  
    avg_acc = correct / total  
  
    return avg_loss, avg_acc
```

---

### （11）单张图片预测与 argparse

项目中还编写了单张图片预测脚本 `predict.py`，用于测试模型对外部图片的识别能力。

命令行参数部分：

```python
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="待预测图片路径"
)
parser.add_argument(
    "--model",
    type=str,
    default="best_catdog_resnet18.pth",
    help="模型权重路径"
)
```

表示运行脚本时，可以这样指定参数：

```bash
python predict.py --image "D:\images\dog.jpg"
```

程序会自动：

1. 加载模型；
    
2. 加载待预测图片；
    
3. 执行与测试集一致的预处理；
    
4. 进行前向传播；
    
5. 输出最终预测类别和置信度。
    


**单张图片预测脚本代码**
```python
def predict_image(image_path: str, model_path: str = "best_catdog_resnet18.pth"):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
    checkpoint = torch.load(  
        model_path,  
        map_location=device,  
        weights_only=False  
    )  
  
    image_size = checkpoint.get("image_size", 224)  
  
    model = CatDogResNet18(  
        num_classes=2,  
        freeze_backbone=True  
    ).to(device)  
  
    model.load_state_dict(checkpoint["model_state_dict"])  
    model.eval()  
  
    transform = get_eval_transform(image_size)  
  
    image_path = Path(image_path)  
  
    if not image_path.exists():  
        raise FileNotFoundError(f"图片不存在: {image_path}")  
  
    image = Image.open(image_path).convert("RGB")  
    image = transform(image)  
    image = image.unsqueeze(0).to(device)  
  
    with torch.no_grad():  
        outputs = model(image)  
        probabilities = torch.softmax(outputs, dim=1)  
        confidence, predicted = torch.max(probabilities, dim=1)  
  
    predicted_label = predicted.item()  
    predicted_name = CLASS_NAMES[predicted_label]  
    confidence_value = confidence.item()  
  
    print(f"预测类别: {predicted_name}")  
    print(f"置信度: {confidence_value:.4f}")  
  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument(  
        "--image",  
        type=str,  
        required=True,  
        help="待预测图片路径"  
    )  
    parser.add_argument(  
        "--model",  
        type=str,  
        default="best_catdog_resnet18.pth",  
        help="模型权重路径"  
    )  
  
    args = parser.parse_args()  
  
    predict_image(  
        image_path=args.image,  
        model_path=args.model  
    )
```

---

## 三、学习过程中重点疑问与理解

本阶段学习中，重点梳理并解决了以下几个疑问：

### 1. 为什么要自定义 Dataset？

因为猫狗数据集不是直接由 PyTorch 内置封装好的标准数据集，而是：

```text
CSV 标签 + 图片文件夹
```

因此需要自己定义：

- 如何根据索引找到图片；
    
- 如何读取标签；
    
- 如何返回模型可用的 `(image, label)`。
    

---

### 2. Dataset 和 DataLoader 有什么区别？

```text
Dataset：负责读取单个样本
DataLoader：负责批量打包样本
```

例如：

```python
train_dataset[0]
```

取出的是：

```text
一张图片 + 一个标签
```

而：

```python
for images, labels in train_loader:
```

取出的是：

```text
一批图片 + 一批标签
```

---

### 3. 为什么训练集划分后，索引仍然能正确找到图片？

因为 `train_test_split()` 只是打乱行顺序，不会改变单行内部的图片名和标签绑定关系。  
而 `reset_index(drop=True)` 只是重新编号，不会改变数据内容。

---

### 4. 为什么训练集可以打乱两次？

一次是：

```python
train_test_split()
```

用于随机划分训练、验证、测试集。

另一次是：

```python
DataLoader(..., shuffle=True)
```

用于每个 epoch 内随机读取训练样本顺序。

这两次打乱作用不同，不会导致图片和标签错乱。

---

### 5. 为什么每个 batch 都要 `optimizer.zero_grad()`？

因为 PyTorch 默认梯度会累积。  
当前训练方式希望：

```text
每个 batch 独立计算梯度并更新参数
```

所以在每次循环开始前，必须清空前一批残留梯度。

---

### 6. 为什么评估阶段要加 `@torch.no_grad()` 和 `model.eval()`？

因为验证和测试不需要训练参数。

- `@torch.no_grad()`：关闭梯度计算，节省显存；
    
- `model.eval()`：关闭 Dropout，并让 BatchNorm 进入评估状态。
    

---

### 7. 模型是怎么知道哪个输出值代表猫、哪个代表狗的？

由训练标签决定：

```text
0 = cat
1 = dog
```

训练会推动：

```text
猫图第 0 个输出更高
狗图第 1 个输出更高
```

预测时再用类别映射表翻译成文字。

---

## 四、后续学习安排

后续将在猫狗分类项目的基础上，正式进入目标检测方向的学习。本周将先完成目标检测入门内容，重点观看相关入门课程，初步理解图像分类与目标检测任务的区别，并学习目标检测中常见的数据集格式与标注方式。
