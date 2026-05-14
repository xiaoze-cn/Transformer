# Transformer 中英机器翻译项目教程

本项目实现了一个简化版的中英机器翻译系统。模型结构参考论文 **Attention Is All You Need** 中提出的 Transformer encoder-decoder 架构，任务是把中文句子翻译成英文句子。

这个项目的目标不是做出商业翻译软件，而是完整复现一个机器翻译实验流程：

```text
选择数据集 -> 预处理数据 -> 训练分词器 -> 搭建 Transformer -> 训练模型 -> 测试 BLEU -> 查看翻译样例
```

项目尽量保持简单，核心代码只有两个文件：

```text
src/
├─ config.py   配置文件
└─ main.py     数据处理、训练、评估、翻译主程序
```

## 1. 数据集选择

本项目使用的数据集是：

```text
IWSLT2017 Chinese-English
```

在代码里的配置是：

```python
dataset_name = "IWSLT/iwslt2017"
dataset_config = "iwslt2017-zh-en"
source_lang = "zh"
target_lang = "en"
```

也就是说，本项目使用的是 **中文到英文** 的翻译方向。

### 1.1 为什么必须选择中英平行语料

这个项目做的是中英机器翻译，所以数据集必须满足一个基本条件：每条样本都要同时包含中文句子和对应的英文翻译。

机器翻译模型不是只学习一种语言本身，而是学习两种语言之间的对应关系。例如：

```text
中文：我喜欢机器翻译。
英文：I like machine translation.
```

这样的成对数据叫做平行语料。模型训练时看到的是大量“中文句子 -> 英文句子”的配对，才能逐渐学会中文表达和英文表达之间的映射关系。

如果数据集只有英文，或者只有中文，都不能直接训练中英翻译模型。如果数据集是英语-德语，也不适合作为本项目的主要数据，因为它学习的是英语和德语之间的对应关系，而不是中文和英文之间的对应关系。

### 1.2 为什么不用原论文的英德数据集

原始 Transformer 论文主要使用了 WMT14 English-German 和 English-French 数据集。这些数据集更大，也更接近论文原始实验。

但是本项目要做的是中英互译方向，直接使用论文里的 English-German 数据集并不合适。原因有两个：

第一，任务语言不一致。English-German 训练出来的是英德翻译模型，不能说明模型完成了中英翻译任务。

第二，课程展示时不直观。项目要求面向中文和英文翻译，如果最后展示的是德语输出，不方便非语言专业同学解释翻译质量，也不方便老师直接判断样例效果。

因此，本项目保留 Transformer 的核心结构思想，但数据集换成中英平行语料。这样既能体现对 Attention Is All You Need 的复现，又能符合中英机器翻译的任务要求。

### 1.3 为什么不用 WMT 中文数据集

也存在更大规模的中英翻译数据集，例如 WMT 系列中的中英数据。但这类数据集的问题是：

- 数据量很大；
- 下载和预处理更麻烦；
- 训练时间更长；
- 对显存和算力要求更高；
- 不适合普通课程作业快速复现。

因此，本项目没有选择更大的 WMT 中英数据，而是选择 IWSLT2017 中英数据。它比 WMT 小很多，但仍然是正式的机器翻译数据集，适合在个人电脑上完成完整实验流程。

### 1.4 为什么不用特别小的数据集

也可以使用 Tatoeba、ManyThings 这类小型中英句对数据集。它们更容易训练，但句子通常很短，任务也比较简单，实验说服力会弱一些。

IWSLT2017 的优点是：

- 规模适中；
- 训练成本可以接受；
- 来源是 TED 演讲翻译，句子比较自然；
- 适合展示 Transformer 翻译模型；
- 比玩具数据集更正式。

### 1.5 当前数据规模

本项目预处理后得到的数据规模是：

```text
train: 231266 sentence pairs
valid: 879 sentence pairs
test: 8549 sentence pairs
```

含义是：

- `train`：训练集，用来更新模型参数；
- `valid`：验证集，用来判断模型训练过程中是否变好；
- `test`：测试集，用来最终评估模型效果。

## 2. 项目目录结构

当前项目结构如下：

```text
Transformer/
├─ README.md
├─ requirements.txt
│
├─ src/
│  ├─ config.py
│  └─ main.py
│
├─ data/
│  ├─ raw/
│  ├─ train.zh
│  ├─ train.en
│  ├─ valid.zh
│  ├─ valid.en
│  ├─ test.zh
│  ├─ test.en
│  ├─ spm.model
│  └─ spm.vocab
│
├─ checkpoints/
│  └─ model.pt
│
└─ outputs/
   ├─ loss.png
   └─ examples.txt
```

### 2.1 src

`src/config.py` 是配置文件，保存数据集、训练参数、模型参数。

`src/main.py` 是主程序，包含：

- 数据预处理；
- Dataset 和 DataLoader；
- Transformer 模型；
- 训练；
- 测试；
- 单句翻译。

### 2.2 data

`data/raw/` 保存 Hugging Face 下载的数据缓存。

`train.zh` 和 `train.en` 是训练集中文和英文。它们按行对应：

```text
train.zh 第 1 行 -> train.en 第 1 行
train.zh 第 2 行 -> train.en 第 2 行
```

`valid.zh / valid.en` 是验证集。

`test.zh / test.en` 是测试集。

`spm.model` 是 SentencePiece 分词器模型。

`spm.vocab` 是 SentencePiece 词表。

### 2.3 checkpoints

`checkpoints/model.pt` 是训练好的 PyTorch 模型参数文件。

训练过程中，如果验证集 loss 下降，就会保存模型：

```python
torch.save(..., "checkpoints/model.pt")
```

### 2.4 outputs

`outputs/loss.png` 是训练 loss 曲线。

`outputs/examples.txt` 保存本次测试结果和翻译样例：

```text
BLEU = 6.39
Test sentences = 8549
```

后面继续保存前 20 条翻译样例：

```text
Source: 中文输入
Reference: 标准英文答案
Prediction: 模型预测英文
```

## 3. 创建 Conda 环境

建议使用单独的 conda 环境，避免和其他项目的 Python 包冲突。

在 PowerShell 中进入项目目录：

```powershell
cd <解压后的项目根目录>
```

创建环境：

```powershell
conda create -n translation python=3.12
```

激活环境：

```powershell
conda activate translation
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 中包含：

```text
torch
datasets
sentencepiece
sacrebleu
matplotlib
tqdm
```

它们分别负责：

- `torch`：深度学习框架；
- `datasets`：下载 Hugging Face 数据集；
- `sentencepiece`：训练和使用分词器；
- `sacrebleu`：计算 BLEU 翻译指标；
- `matplotlib`：绘制 loss 曲线；
- `tqdm`：显示训练进度条。

## 4. 运行流程

### 4.1 数据预处理

```powershell
python src/main.py preprocess
```

这一步会：

1. 下载 IWSLT2017 中英数据集；
2. 生成 `train.zh / train.en`；
3. 生成 `valid.zh / valid.en`；
4. 生成 `test.zh / test.en`；
5. 训练 SentencePiece 分词器；
6. 生成 `spm.model / spm.vocab`。

如果数据已经下载过，Hugging Face 会优先使用本地缓存。

### 4.2 训练模型

```powershell
python src/main.py train
```

这一步会：

1. 加载分词器；
2. 读取训练集和验证集；
3. 创建 Transformer 模型；
4. 训练 20 个 epoch；
5. 保存验证集效果最好的模型到 `checkpoints/model.pt`；
6. 保存 loss 曲线到 `outputs/loss.png`。

### 4.3 评估模型

```powershell
python src/main.py evaluate
```

这一步会：

1. 加载 `checkpoints/model.pt`；
2. 遍历测试集中文句子；
3. 调用模型生成英文翻译；
4. 和标准英文答案对比；
5. 计算 BLEU；
6. 打印前 20 条翻译样例；
7. 保存 BLEU、测试句子数量和前 20 条样例到 `outputs/examples.txt`。

当前一次实验结果是：

```text
BLEU = 6.39
Test sentences = 8549
```

这个分数说明模型已经学到了一些翻译能力，但效果还比较基础。它可以翻出简单词义和部分短句，但长句会出现漏译、重复和语义偏移。

### 4.4 单句翻译

```powershell
python src/main.py translate --text "我喜欢机器翻译。"
```

当前模型输出示例：

```text
I love machines.
```

这个结果说明模型学到了“我喜欢”和“机器”的对应关系，但漏掉了“翻译”。因此它不是随机输出，但训练效果还不够强。

## 5. 配置文件讲解

配置文件是 `src/config.py`。

它使用 Python 的 `dataclass` 写法，把不同类型的配置分开。

### 5.1 数据配置

```python
@dataclass
class DataConfig:
    dataset_name: str = "IWSLT/iwslt2017"
    dataset_config: str = "iwslt2017-zh-en"
    source_lang: str = "zh"
    target_lang: str = "en"
    raw_dir: str = "data/raw"
    data_dir: str = "data"
    tokenizer_prefix: str = "data/spm"
    vocab_size: int = 16000
    max_length: int = 96
```

解释：

- `dataset_name`：数据集名称；
- `dataset_config`：数据集子配置，这里是中英；
- `source_lang`：源语言，中文；
- `target_lang`：目标语言，英文；
- `raw_dir`：原始数据缓存路径；
- `data_dir`：处理后的数据路径；
- `tokenizer_prefix`：分词器保存路径前缀；
- `vocab_size`：词表大小；
- `max_length`：句子最大 token 长度。

### 5.2 训练配置

```python
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0005
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    device: str = "auto"
```

解释：

- `epochs`：完整遍历训练集的次数；
- `batch_size`：每次送入模型的句子数量；
- `learning_rate`：学习率，控制参数更新幅度；
- `label_smoothing`：标签平滑，让模型不要过度自信；
- `grad_clip`：梯度裁剪，防止训练不稳定；
- `device`：`auto` 表示有 GPU 就用 GPU。

### 5.3 模型配置

```python
@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
```

解释：

- `d_model`：每个 token 的向量维度；
- `nhead`：多头注意力的头数；
- `num_layers`：encoder 和 decoder 的层数；
- `dim_feedforward`：前馈网络隐藏层维度；
- `dropout`：随机丢弃部分神经元，防止过拟合。

### 5.4 解码配置

```python
@dataclass
class DecodeConfig:
    max_length: int = 96
    beam_size: int = 4
```

当前代码实际使用的是 greedy decoding，也就是每一步都选择概率最高的词。

`beam_size` 目前保留在配置里，方便后续升级成 beam search。

## 6. main.py 代码分段讲解

这一章按代码片段来讲，不按单行拆解。每一段先看几行代码，再说明它在完整流程中的作用。这样更容易理解上下文：这段代码属于数据处理、模型定义、训练，还是评估。

`main.py` 的整体顺序是：

```text
导入库和配置 -> 定义特殊符号 -> 数据预处理 -> 数据集读取 -> batch 补齐 -> Transformer 模型 -> 训练 -> 翻译 -> 评估 -> 命令行入口
```

### 6.1 导入库和配置

```python
from pathlib import Path
import argparse
import math
import random

import matplotlib.pyplot as plt
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import cfg
```

这一段是程序准备阶段。`Path` 用来处理文件路径，`argparse` 用来解析命令行参数，`math` 用于位置编码和 embedding 缩放，`random` 用于设置随机种子。

深度学习部分主要依赖 PyTorch，也就是 `torch` 和 `torch.nn`。`datasets` 用来下载 Hugging Face 上的 IWSLT2017 数据集。`sentencepiece` 用来训练和加载分词器。`sacrebleu` 用来计算 BLEU 分数。`matplotlib` 用来画 loss 曲线。`tqdm` 用来显示训练进度条。

最后一行 `from config import cfg` 很重要。项目里的参数不直接散落在主程序中，而是统一放在 `config.py`，例如数据集名称、训练轮数、batch size、模型维度等。这样主程序负责流程，配置文件负责参数。

### 6.2 特殊 token 和设备选择

```python
PAD = 0
BOS = 1
EOS = 2


def device():
    name = cfg.train.device
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
```

这段代码定义了三个特殊 token。`PAD` 用来补齐短句，`BOS` 表示句子开始，`EOS` 表示句子结束。

翻译模型处理的是数字序列，不是原始文字。比如一句英文目标句会被组织成：

```text
[BOS, I, like, machine, translation, EOS]
```

`BOS` 告诉 decoder 从哪里开始生成，`EOS` 告诉 decoder 什么时候结束。`PAD` 用来把一个 batch 里的短句补到同样长度。

`device()` 函数决定使用 GPU 还是 CPU。配置里写的是 `auto`，所以如果当前环境能使用 CUDA，就返回 `cuda`；否则返回 `cpu`。这样同一份代码可以在不同电脑上运行。

### 6.3 preprocess：加载数据集

```python
def preprocess():
    data = cfg.data
    Path(data.data_dir).mkdir(parents=True, exist_ok=True)
    Path(data.raw_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        data.dataset_name,
        data.dataset_config,
        cache_dir=data.raw_dir,
        trust_remote_code=True,
    )
```

`preprocess()` 是数据预处理入口，对应命令：

```powershell
python src/main.py preprocess
```

这一段先读取数据配置，然后创建 `data/` 和 `data/raw/` 两个目录。`data/` 保存处理后的文本文件和分词器，`data/raw/` 保存 Hugging Face 下载缓存。

`load_dataset()` 会加载 IWSLT2017 中英数据集。配置中 `dataset_config = "iwslt2017-zh-en"`，所以这里加载的是中文和英文配对的平行语料。如果本地没有缓存，它会下载；如果已经下载过，通常会复用本地缓存。

### 6.4 preprocess：生成 train/valid/test 文本文件

```python
    splits = {"train": "train", "validation": "valid", "test": "test"}
    tok_path = Path(data.data_dir) / "tokenizer_train.txt"

    with tok_path.open("w", encoding="utf-8") as tok_f:
        for hf_name, out_name in splits.items():
            src_file = Path(data.data_dir) / f"{out_name}.{data.source_lang}"
            tgt_file = Path(data.data_dir) / f"{out_name}.{data.target_lang}"
            count = 0
```

这一段准备把 Hugging Face 数据集转换成普通文本文件。Hugging Face 中验证集叫 `validation`，项目里保存为 `valid`，所以最后会生成：

```text
data/train.zh    data/train.en
data/valid.zh    data/valid.en
data/test.zh     data/test.en
```

`.zh` 文件保存中文，`.en` 文件保存英文。两边按行对应：中文文件第 N 行对应英文文件第 N 行。

`tokenizer_train.txt` 是专门给 SentencePiece 训练分词器用的文本，它会把训练集里的中文和英文都写进去。

### 6.5 preprocess：逐条写入中英句对

```python
            with src_file.open("w", encoding="utf-8") as src_f, tgt_file.open("w", encoding="utf-8") as tgt_f:
                for row in ds[hf_name]:
                    pair = row["translation"]
                    src = " ".join(pair[data.source_lang].strip().split())
                    tgt = " ".join(pair[data.target_lang].strip().split())
                    if not src or not tgt:
                        continue
                    src_f.write(src + "\n")
                    tgt_f.write(tgt + "\n")
                    if out_name == "train":
                        tok_f.write(src + "\n" + tgt + "\n")
                    count += 1
            print(f"{out_name}: {count} sentence pairs")
```

这一段是真正写文件的地方。每条数据有一个 `translation` 字段，里面同时包含中文和英文。代码取出中文作为 `src`，英文作为 `tgt`。

`strip().split()` 再重新拼接，是为了清理多余空格，让文本格式统一。如果某条样本中文或英文为空，就跳过。

中文写入 `.zh` 文件，英文写入 `.en` 文件。训练集还会额外写入 `tokenizer_train.txt`，用于后续训练分词器。验证集和测试集不参与分词器训练，这样更符合机器学习实验规范。

### 6.6 preprocess：训练 SentencePiece 分词器

```python
    spm.SentencePieceTrainer.train(
        input=str(tok_path),
        model_prefix=data.tokenizer_prefix,
        vocab_size=data.vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        unk_id=3,
    )
```

神经网络不能直接处理文字，它需要数字。SentencePiece 的作用就是把中文和英文句子切成 token，并把 token 转成 id。

这里使用 BPE 分词。BPE 会把常见字符片段合并成子词，既能处理常见词，也能处理没见过的新词。`vocab_size = 16000` 表示词表大小是 16000。

`pad_id / bos_id / eos_id / unk_id` 保证分词器中的特殊符号和程序里的 `PAD / BOS / EOS` 对应一致。

### 6.7 TranslationData：读取文本并变成数字序列

```python
class TranslationData(Dataset):
    def __init__(self, split, sp):
        data = cfg.data
        src_path = Path(data.data_dir) / f"{split}.{data.source_lang}"
        tgt_path = Path(data.data_dir) / f"{split}.{data.target_lang}"
        max_len = data.max_length
        self.items = []
```

`TranslationData` 是 PyTorch 数据集类。它根据 `split` 决定读取训练集、验证集还是测试集。

例如 `split="train"` 时，它会读取：

```text
data/train.zh
data/train.en
```

`sp` 是 SentencePiece 分词器。后面会用它把文本句子转换成 token id。

```python
        with src_path.open("r", encoding="utf-8") as src_f, tgt_path.open("r", encoding="utf-8") as tgt_f:
            for src, tgt in zip(src_f, tgt_f):
                src_ids = [BOS] + sp.encode(src.strip(), out_type=int)[: max_len - 2] + [EOS]
                tgt_ids = [BOS] + sp.encode(tgt.strip(), out_type=int)[: max_len - 2] + [EOS]
                self.items.append((src_ids, tgt_ids))
```

这一段逐行读取中英句对。`sp.encode(..., out_type=int)` 会把句子变成数字 id。前面加 `BOS`，后面加 `EOS`，表示句子开始和结束。

`[: max_len - 2]` 是长度截断，因为还要留两个位置给 BOS 和 EOS。这样可以防止极长句子占用过多显存。

### 6.8 batchify：把不同长度的句子补齐

```python
def batchify(batch):
    srcs, tgts = zip(*batch)
    src = torch.full((len(batch), max(map(len, srcs))), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), max(map(len, tgts))), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt
```

一个 batch 中的句子长短不同，但模型需要矩阵形式的输入。所以这段代码先创建全是 `PAD` 的矩阵，再把真实 token id 填进去。

例如一个 batch 里最长句子长度是 10，那么短句也会被补到长度 10。补出来的位置是 `PAD`，后续模型会通过 mask 忽略这些位置。

最终返回的 `src` 和 `tgt` 都是二维张量：

```text
batch_size x sequence_length
```

### 6.9 PositionalEncoding：给词向量加入顺序信息

```python
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0))
```

Transformer 的注意力机制可以同时看整句话，但它本身不知道词的顺序。如果不给位置信息，模型很难区分“我喜欢你”和“你喜欢我”。

所以这里构造了 sin/cos 位置编码。每个位置都有一个固定向量，后面会加到 token embedding 上。

`register_buffer` 表示 `pe` 是模型的一部分，会跟着模型移动到 GPU，但它不是需要训练的参数。

```python
    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])
```

这段表示把位置编码加到输入向量上，再做 dropout。

### 6.10 Translator：搭建 Transformer 翻译模型

```python
class Translator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        model_cfg = cfg.model
        dim = model_cfg.d_model
        self.dim = dim
        self.src_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = PositionalEncoding(dim, model_cfg.dropout)
```

`Translator` 是本项目的核心模型。

`src_emb` 把中文 token id 转成向量，`tgt_emb` 把英文 token id 转成向量。虽然中英文共用同一个 SentencePiece 词表，但 encoder 和 decoder 分别有自己的 embedding 层。

`self.pos` 是位置编码。token embedding 加上位置编码以后，模型既知道 token 是什么，也知道 token 在句子中的位置。

```python
        self.net = nn.Transformer(
            d_model=dim,
            nhead=model_cfg.nhead,
            num_encoder_layers=model_cfg.num_layers,
            num_decoder_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
            batch_first=True,
        )
        self.out = nn.Linear(dim, vocab_size)
```

`nn.Transformer` 是 PyTorch 封装好的 encoder-decoder Transformer。这里设置模型维度、注意力头数、encoder/decoder 层数、前馈网络维度和 dropout。

`self.out` 是输出层。Transformer 输出的是隐藏向量，但翻译任务需要预测词表中的下一个 token，所以要用线性层把隐藏向量映射到 `vocab_size` 维。

### 6.11 forward：模型如何从输入得到输出

```python
    def forward(self, src, tgt):
        src_pad = src.eq(PAD)
        tgt_pad = tgt.eq(PAD)
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1)
        src = self.pos(self.src_emb(src) * math.sqrt(self.dim))
        tgt = self.pos(self.tgt_emb(tgt) * math.sqrt(self.dim))
```

`forward` 是模型的一次前向传播。

`src_pad` 和 `tgt_pad` 是 padding mask，用来告诉 Transformer 哪些位置是补齐出来的，不应该参与注意力计算。

`tgt_mask` 是 decoder 的未来信息遮罩。训练时虽然完整英文答案已经在数据里，但模型预测当前位置时不能偷看后面的词，否则就相当于作弊。

后两行把 token id 转成 embedding，乘以 `sqrt(dim)` 做缩放，再加位置编码。

```python
        x = self.net(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(x)
```

这段把中文输入和英文前缀送入 Transformer。encoder 理解中文句子，decoder 根据已经看到的英文前缀和 encoder 输出预测后续英文。

最后 `self.out(x)` 输出每个位置上对整个词表的预测分数。

### 6.12 train_epoch：训练一轮数据

```python
def train_epoch(model, loader, loss_fn, optim, dev):
    model.train()
    total_loss = total_tokens = 0
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(dev), tgt.to(dev)
        pred = model(src, tgt[:, :-1])
        gold = tgt[:, 1:]
```

`train_epoch` 表示训练完整一遍训练集。

`tgt[:, :-1]` 是输入给 decoder 的英文前缀，`tgt[:, 1:]` 是模型需要预测的真实答案。例如：

```text
目标句:    [BOS, I, like, machine, EOS]
decoder输入:[BOS, I, like, machine]
模型答案:  [I,   like, machine, EOS]
```

这种训练方式叫 teacher forcing。

```python
        loss = loss_fn(pred.reshape(-1, pred.size(-1)), gold.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optim.step()
```

这一段是标准训练步骤。先计算交叉熵 loss，再清空旧梯度，然后反向传播，接着做梯度裁剪，最后用优化器更新模型参数。

```python
        tokens = gold.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens
```

这里统计平均 loss 时只计算非 `PAD` 的真实 token。padding 位置只是为了补齐矩阵，不应该影响训练指标。

### 6.13 valid_epoch：验证集评估

```python
@torch.no_grad()
def valid_epoch(model, loader, loss_fn, dev):
    model.eval()
    total_loss = total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(dev), tgt.to(dev)
        pred = model(src, tgt[:, :-1])
        gold = tgt[:, 1:]
        loss = loss_fn(pred.reshape(-1, pred.size(-1)), gold.reshape(-1))
```

验证函数和训练函数很像，但它不更新参数。

`@torch.no_grad()` 表示不计算梯度，可以节省显存和时间。`model.eval()` 会关闭 dropout 等训练时才使用的行为，让验证结果更稳定。

验证 loss 用来判断模型是否变好。如果验证 loss 更低，说明模型在未参与训练的数据上表现更好。

### 6.14 train：完整训练流程

```python
def train():
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = device()
    sp = load_sp()
    model = Translator(sp.get_piece_size()).to(dev)
    train_loader = DataLoader(TranslationData("train", sp), batch_size=cfg.train.batch_size, shuffle=True, collate_fn=batchify)
    valid_loader = DataLoader(TranslationData("valid", sp), batch_size=cfg.train.batch_size, collate_fn=batchify)
```

`train()` 是完整训练入口，对应命令：

```powershell
python src/main.py train
```

它先设置随机种子，再选择设备，加载分词器，创建模型，然后创建训练集和验证集的 DataLoader。

训练集设置 `shuffle=True`，表示每个 epoch 打乱训练样本顺序。验证集不需要打乱。

```python
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.train.label_smoothing)
```

优化器使用 Adam。损失函数是交叉熵。`ignore_index=PAD` 表示忽略 padding 位置，`label_smoothing` 可以让模型不要过度自信。

```python
    for epoch in range(1, cfg.train.epochs + 1):
        tr = train_epoch(model, train_loader, loss_fn, optim, dev)
        va = valid_epoch(model, valid_loader, loss_fn, dev)
        history["train"].append(tr)
        history["valid"].append(va)
        print(f"epoch {epoch:02d}: train={tr:.4f}, valid={va:.4f}")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "vocab_size": sp.get_piece_size()}, "checkpoints/model.pt")
```

每个 epoch 都先训练，再验证。如果验证 loss 比之前更低，就保存模型到 `checkpoints/model.pt`。也就是说，保存的是验证集上表现最好的模型，而不一定是最后一轮模型。

训练结束后，代码会把训练 loss 和验证 loss 画成曲线，保存为：

```text
outputs/loss.png
```

### 6.15 翻译一句话

```python
def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Translator(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    return model
```

这段负责加载训练好的模型。保存时存了模型参数和词表大小，加载时先按词表大小重新创建模型，再把参数填进去。

```python
@torch.no_grad()
def translate_sentence(model, sp, text, dev):
    ids = [BOS] + sp.encode(text.strip(), out_type=int)[: cfg.data.max_length - 2] + [EOS]
    src = torch.tensor([ids], device=dev)
    tgt = torch.tensor([[BOS]], device=dev)
```

翻译时先把输入中文转成 token id，并加上 BOS 和 EOS。decoder 的初始输入只有 `[BOS]`，后面的英文由模型一步一步生成。

```python
    model.eval()
    for _ in range(cfg.decode.max_length - 1):
        next_id = int(model(src, tgt)[:, -1].argmax(-1).item())
        tgt = torch.cat([tgt, torch.tensor([[next_id]], device=dev)], dim=1)
        if next_id == EOS:
            break
    return sp.decode([i for i in tgt[0].tolist() if i not in (PAD, BOS, EOS)])
```

这段是自回归生成。模型每次预测下一个 token，并把预测结果接到 `tgt` 后面。下一轮再根据更长的 `tgt` 继续预测。

`argmax` 表示每一步都选概率最高的 token，这叫 greedy decoding。生成到 `EOS` 就停止。最后去掉特殊 token，用 SentencePiece 解码成英文文本。

### 6.16 evaluate：计算 BLEU 并保存样例

```python
def evaluate(ckpt, limit=None):
    dev = device()
    sp = load_sp()
    model = load_checkpoint(ckpt).to(dev)
    src_path = Path(cfg.data.data_dir) / f"test.{cfg.data.source_lang}"
    tgt_path = Path(cfg.data.data_dir) / f"test.{cfg.data.target_lang}"
    preds, refs, examples = [], [], []
```

`evaluate()` 是测试入口，对应命令：

```powershell
python src/main.py evaluate
```

它加载模型和分词器，然后读取测试集 `test.zh` 和 `test.en`。`preds` 保存模型翻译，`refs` 保存标准答案，`examples` 保存前 20 条样例。

```python
    with src_path.open("r", encoding="utf-8") as src_f, tgt_path.open("r", encoding="utf-8") as tgt_f:
        for i, (src, ref) in enumerate(zip(src_f, tgt_f)):
            if limit is not None and i >= limit:
                break
            src, ref = src.strip(), ref.strip()
            pred = translate_sentence(model, sp, src, dev)
            preds.append(pred)
            refs.append(ref)
            if len(examples) < 20:
                examples.append((src, ref, pred))
```

这一段逐句翻译测试集。每一句中文都会得到一个英文预测。`--limit` 可以限制评估数量，比如只测前 100 条，方便快速检查。

```python
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/examples.txt", "w", encoding="utf-8") as f:
        for src, ref, pred in examples:
            f.write(f"Source: {src}\nReference: {ref}\nPrediction: {pred}\n\n")
```

`sacrebleu` 用来计算 BLEU 分数。BLEU 越高，说明模型翻译和参考答案越接近。

前 20 条样例会保存到 `outputs/examples.txt`。报告里可以引用这些例子分析模型翻译效果。

### 6.17 main：命令行入口

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preprocess", "train", "evaluate", "translate"])
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    parser.add_argument("--text")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
```

`main()` 是命令行入口。它读取用户输入的命令，并判断要执行哪个功能。

支持四个命令：

```text
preprocess  数据预处理
train       训练模型
evaluate    测试模型
translate   单句翻译
```

```python
    if args.command == "preprocess":
        preprocess()
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate(args.ckpt, args.limit)
    elif args.command == "translate":
        if not args.text:
            raise SystemExit("--text is required")
        dev = device()
        print(translate_sentence(load_checkpoint(args.ckpt).to(dev), load_sp(), args.text, dev))
```

这一段把命令分发到对应函数。例如：

```powershell
python src/main.py train
```

会调用 `train()`。

```powershell
python src/main.py translate --text "我喜欢机器翻译。"
```

会加载模型和分词器，然后翻译这一句话。

所以 `main.py` 可以理解为一个统一入口。它把数据处理、训练、评估、翻译四个步骤都放在一个脚本里，通过命令参数切换功能。

### 6.18 把 main.py 的数据流完整串起来

前面的讲解是按代码片段拆开的，这里再从“数据怎么一步步流动”的角度把程序串起来。

第一步，原始数据来自 Hugging Face：

```python
ds = load_dataset(...)
```

这时数据还不是普通文本文件，而是 Hugging Face Dataset 对象。每条样本大致可以理解成：

```python
{
    "translation": {
        "zh": "中文句子",
        "en": "English sentence"
    }
}
```

第二步，`preprocess()` 把这些样本写成普通文本文件：

```text
data/train.zh
data/train.en
data/valid.zh
data/valid.en
data/test.zh
data/test.en
```

这一步的意义是把数据整理成最直观、最容易检查的格式。打开 `train.zh` 可以看到中文句子，打开 `train.en` 可以看到对应英文句子。两个文件按行对齐。

第三步，SentencePiece 根据训练集生成分词器：

```text
data/spm.model
data/spm.vocab
```

`spm.model` 是程序真正使用的分词器模型。后面训练、评估、翻译都必须用同一个 `spm.model`。如果训练和预测使用不同分词器，token id 会对不上，模型就无法正常工作。

第四步，`TranslationData` 读取文本文件，并把文字转成数字：

```python
src_ids = [BOS] + sp.encode(src) + [EOS]
tgt_ids = [BOS] + sp.encode(tgt) + [EOS]
```

也就是说，原始文本会变成类似这样的数字序列：

```text
中文：我喜欢机器翻译。
数字：[1, 234, 567, 891, 2]
```

这里的 `1` 是 BOS，`2` 是 EOS。

第五步，`batchify()` 把多个样本组成一个 batch：

```text
单条样本：一个中文数字序列 + 一个英文数字序列
一个 batch：多条样本组成的矩阵
```

因为句子长短不同，所以 batchify 会用 `PAD` 补齐。补齐之后，模型才能一次处理多句话。

第六步，训练时模型接收两个输入：

```python
pred = model(src, tgt[:, :-1])
```

其中：

```text
src         中文句子
tgt[:, :-1] 英文答案去掉最后一个 token，作为 decoder 输入
```

模型输出 `pred`，表示每个位置上对下一个英文 token 的预测。

第七步，loss 计算时使用：

```python
gold = tgt[:, 1:]
```

也就是说，模型看到：

```text
[BOS, I, like, machine]
```

应该预测：

```text
[I, like, machine, EOS]
```

这就是机器翻译训练中常见的 teacher forcing。

第八步，如果验证集 loss 变低，就保存模型：

```python
torch.save(..., "checkpoints/model.pt")
```

这个文件保存的是模型学到的参数。后面 `evaluate` 和 `translate` 都会加载它。

第九步，评估时程序读取测试集：

```text
data/test.zh
data/test.en
```

对每一句中文调用：

```python
translate_sentence(...)
```

生成英文预测，然后和标准英文 `Reference` 比较，计算 BLEU。

所以整条数据流可以总结成：

```text
Hugging Face 数据集
-> train/valid/test 文本文件
-> SentencePiece token id
-> batch 矩阵
-> Transformer
-> loss / model.pt
-> BLEU / examples.txt
```

### 6.19 关键变量的含义和形状

理解深度学习代码时，除了知道函数做什么，还要知道变量大概长什么样。下面用一个 batch 来解释。

假设 batch size 是 32，当前 batch 里最长中文句子长度是 60，最长英文句子长度是 55。那么：

```python
src.shape
```

大致是：

```text
[32, 60]
```

含义是：

```text
32 句话，每句话补齐到 60 个 token
```

同理：

```python
tgt.shape
```

大致是：

```text
[32, 55]
```

训练时 decoder 输入是：

```python
tgt[:, :-1]
```

形状会变成：

```text
[32, 54]
```

标准答案是：

```python
tgt[:, 1:]
```

形状也是：

```text
[32, 54]
```

模型输出：

```python
pred = model(src, tgt[:, :-1])
```

形状大致是：

```text
[32, 54, 16000]
```

这里的 `16000` 是词表大小。也就是说，对 batch 中每一句话、每一个输出位置，模型都会给出 16000 个 token 的预测分数。

例如某个位置可能是在预测英文句子的第 3 个 token。模型不会直接输出一个单词，而是输出整个词表的分数：

```text
I: 0.01
like: 0.20
machine: 0.35
translation: 0.10
...
```

训练时，交叉熵 loss 会比较模型预测分数和标准答案 token，告诉模型哪里预测错了。

### 6.20 关键文件和代码之间的对应关系

为了更容易讲项目，可以记住下面这张对应关系。

```text
src/config.py
```

保存项目参数，例如训练轮数、batch size、模型大小、数据路径。

```text
src/main.py preprocess
```

生成：

```text
data/train.zh
data/train.en
data/valid.zh
data/valid.en
data/test.zh
data/test.en
data/spm.model
data/spm.vocab
```

```text
src/main.py train
```

读取：

```text
data/train.zh
data/train.en
data/valid.zh
data/valid.en
data/spm.model
```

生成：

```text
checkpoints/model.pt
outputs/loss.png
```

```text
src/main.py evaluate
```

读取：

```text
checkpoints/model.pt
data/test.zh
data/test.en
data/spm.model
```

生成：

```text
outputs/examples.txt
```

并在终端打印：

```text
BLEU = ...
Test sentences = ...
```

```text
src/main.py translate --text "..."
```

读取：

```text
checkpoints/model.pt
data/spm.model
```

输出：

```text
一条英文翻译
```

### 6.21 这份代码为什么这样简化

这个项目没有把代码拆成很多文件，例如 `model.py`、`dataset.py`、`train.py`、`evaluate.py`。原因是课程项目更需要清晰展示完整流程，而不是做一个复杂工程。

因此代码只保留两个核心文件：

```text
config.py  参数
main.py    流程和模型
```

这样优点是：

- 文件少，容易交付；
- 跑法简单，命令统一；
- 非计算机专业同学也能从上到下读完；
- 报告里解释项目结构更方便。

缺点是：

- `main.py` 会比较长；
- 工程化程度不如大型项目；
- 后续扩展 beam search 或更多模型时，可能需要重新拆文件。

但对于本课程项目来说，这种结构是合适的。它保留了机器翻译实验最重要的部分：数据、分词、模型、训练、评估和样例展示。

## 7. Transformer 原理详细说明

Transformer 是一种处理序列数据的神经网络结构。它最早在机器翻译任务中提出，后来成为大语言模型的基础。

在机器翻译里，可以把 Transformer 理解成两个部分：

```text
Encoder：读懂中文
Decoder：根据中文含义生成英文
```

它和传统模型最大的区别是：Transformer 不靠循环结构一个词一个词传递信息，而是用 attention 让句子里的词直接互相建立联系。

### 7.1 传统 RNN 的问题

在 Transformer 之前，常见的序列模型是 RNN、LSTM、GRU。

它们处理句子的方式是：

```text
第 1 个词 -> 第 2 个词 -> 第 3 个词 -> ...
```

这种方式的问题是：

- 不能很好地并行计算；
- 长句信息容易丢失；
- 训练速度慢；
- 远距离词之间关系难学。

例如：

```text
The book that I bought yesterday is interesting.
```

`book` 和 `is` 相隔较远，但语法上有关联。RNN 需要一步一步传递信息，长距离关系容易变弱。

### 7.2 Transformer 的核心思想

Transformer 的核心是 attention。

Attention 的直观含义是：

```text
在理解一个词时，模型应该关注句子里的哪些其他词？
```

例如：

```text
我喜欢机器翻译。
```

当模型翻译“翻译”时，它应该关注：

```text
机器
翻译
```

而不是只看当前这个词。

Attention 允许模型直接建立任意两个词之间的联系，不需要一步一步传递。

这对翻译很重要。因为翻译时，一个词的意思往往要结合远处的词才能确定。

例如：

```text
我 把 书 放 在 桌子 上
```

翻译 `上` 时，模型需要知道它和 `桌子` 有关，而不是单独把 `上` 翻成 `up`。Attention 可以让 `上` 直接关注 `桌子`，从而更容易翻译成：

```text
on the table
```

### 7.3 Self-Attention

Self-attention 是 Transformer encoder 的核心。

它的意思是：

```text
一个句子内部的词互相看彼此
```

例如中文输入：

```text
我 喜欢 机器 翻译
```

模型会计算：

```text
我     应该关注哪些词
喜欢   应该关注哪些词
机器   应该关注哪些词
翻译   应该关注哪些词
```

每个词都会根据其他词的信息更新自己的表示。

更新后的表示不再是孤立的词向量。比如 `机器` 原本只是一个词，但经过 self-attention 后，它的表示会融合周围词的信息，知道它出现在 `机器翻译` 这个短语里，而不是单独指一台机器。

### 7.4 Query、Key、Value

Attention 常用三个概念：

```text
Query
Key
Value
```

可以用一个生活化比喻理解：

```text
Query：我正在寻找什么信息
Key：每个词提供的索引
Value：每个词真正包含的信息
```

Attention 会用 Query 和 Key 计算相关性，再根据相关性加权 Value。

可以把它想成一个检索过程。

假设模型正在处理 `翻译` 这个 token：

```text
Query：我现在处理的是“翻译”，我需要找哪些上下文？
Key：句子中每个 token 都给出一个“我是什么”的索引
Value：每个 token 真正携带的信息
```

模型会拿 `翻译` 的 Query 去和所有 token 的 Key 比较，得到一组分数。分数越高，说明越应该关注那个 token。最后再按照这些分数把 Value 加权求和，得到新的 `翻译` 表示。

数学上可以写成：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

不用死记公式，可以理解为：

```text
先算相关程度，再按相关程度汇总信息
```

### 7.5 Multi-Head Attention

一个 attention 头只能从一个角度看句子。

Multi-head attention 的意思是：

```text
多个 attention 头同时从不同角度看句子
```

有的头可能关注语法关系，有的头可能关注词义关系，有的头可能关注位置关系。

本项目配置是：

```python
nhead = 4
```

也就是使用 4 个 attention head。

为什么需要多个头？因为一句话里同时存在多种关系。

还是以：

```text
我 喜欢 机器 翻译
```

为例，不同 attention head 可能学习到不同关注方式：

```text
head 1：关注主语和动作，比如“我”和“喜欢”
head 2：关注短语组合，比如“机器”和“翻译”
head 3：关注位置相邻关系
head 4：关注整体语义
```

这些头的结果会合并起来，形成更丰富的句子表示。

### 7.6 Encoder 和 Decoder

机器翻译使用的是 encoder-decoder Transformer。

Encoder 负责理解源语言：

```text
中文句子 -> 语义表示
```

Decoder 负责生成目标语言：

```text
语义表示 + 已经生成的英文 -> 下一个英文词
```

本项目中：

```text
Encoder 输入中文
Decoder 输出英文
```

更完整地说，训练时模型输入有两部分：

```text
Encoder 输入：中文句子
Decoder 输入：标准英文答案的前半部分
```

模型要预测：

```text
标准英文答案的后半部分
```

例如目标英文是：

```text
[BOS, I, like, machine, translation, EOS]
```

训练时 decoder 输入：

```text
[BOS, I, like, machine, translation]
```

模型预测目标：

```text
[I, like, machine, translation, EOS]
```

这就是前面代码里：

```python
pred = model(src, tgt[:, :-1])
gold = tgt[:, 1:]
```

的含义。

### 7.7 Encoder 做什么

Encoder 读取完整中文句子，通过 self-attention 建立中文词之间的关系。

例如：

```text
我 喜欢 机器 翻译
```

Encoder 输出的是每个中文 token 的上下文表示。

这些表示不再只是单个词的意思，而是结合了整句话上下文的信息。

Encoder 内部每一层大致包含：

```text
self-attention -> feed forward
```

self-attention 让中文 token 之间互相交换信息。feed forward 再对每个位置的表示进行进一步加工。

多层 encoder 叠加之后，模型对中文句子的理解会逐步加深。第一层可能更多学习局部词语关系，后面的层可能学习更抽象的语义关系。

### 7.8 Decoder 做什么

Decoder 生成英文句子。

它有两种 attention：

第一种是 masked self-attention：

```text
看已经生成的英文词
```

第二种是 cross-attention：

```text
看 encoder 输出的中文信息
```

例如要生成：

```text
I like machine translation
```

当 decoder 正在生成 `translation` 时，它可以看：

```text
已经生成的英文：I like machine
中文输入：我喜欢机器翻译
```

Decoder 的工作比 encoder 更复杂，因为它既要保证英文句子自身通顺，又要保证英文内容对应中文输入。

所以 decoder 有三类信息来源：

```text
1. 当前已经生成的英文
2. encoder 对中文句子的理解
3. 位置信息
```

第一类信息帮助它生成语法通顺的英文。第二类信息帮助它不要偏离中文原意。第三类信息帮助它知道当前生成到句子的哪个位置。

### 7.9 为什么 decoder 要 mask

训练时，完整英文答案是已知的。

如果不加 mask，模型在预测第 3 个词时可能提前看到第 4 个词，这就相当于作弊。

所以 decoder 要用 mask 保证：

```text
预测当前位置时，只能看当前位置之前的词
```

本项目中对应代码是：

```python
tgt_mask = torch.triu(..., diagonal=1)
```

这个 mask 可以理解成一个上三角矩阵。它会把当前位置右边的未来 token 遮住。

例如英文目标句是：

```text
BOS I like machine translation EOS
```

当模型预测 `like` 时，它可以看到：

```text
BOS I
```

但不能看到：

```text
machine translation EOS
```

这样训练方式才和真正翻译时一致。因为真正翻译时，未来词还没有生成出来。

### 7.10 Feed Forward Network

Transformer 每层除了 attention，还有前馈网络。

Attention 负责汇总上下文信息。

Feed Forward Network 负责对每个位置的表示做进一步非线性变换。

可以理解成：

```text
Attention 负责看别人
Feed Forward 负责加工自己
```

更具体地说，attention 负责“信息交换”，feed forward 负责“特征变换”。每个 token 先通过 attention 得到上下文信息，然后通过前馈网络把这些信息加工成更适合下一层使用的表示。

### 7.11 Residual 和 LayerNorm

原始 Transformer 中还有残差连接和 Layer Normalization。

本项目使用 PyTorch 的 `nn.Transformer`，这些内部已经封装好了。

它们的作用是：

- 让深层网络更容易训练；
- 避免梯度消失；
- 稳定训练过程。

残差连接可以理解成给信息开了一条捷径。即使某一层学得不好，原始信息也可以绕过去继续传递。

LayerNorm 则像是把每一层的数值范围整理一下，避免中间结果忽大忽小，导致训练不稳定。

### 7.12 Positional Encoding 为什么重要

Transformer 本身没有顺序概念。

如果没有 positional encoding，模型看到：

```text
我 喜欢 你
```

和：

```text
你 喜欢 我
```

可能很难区分顺序差异。

所以需要给每个位置加一个固定的位置向量。

本项目使用的是经典的 sin/cos 位置编码。

sin/cos 位置编码不是训练出来的，而是按公式生成的固定向量。它的好处是每个位置都有独特编码，而且不同位置之间有规律可循，模型可以学习相对位置关系。

本项目对应代码是：

```python
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
```

其中偶数维使用 sin，奇数维使用 cos。

### 7.13 本项目和原始论文的关系

原始论文 **Attention Is All You Need** 提出了完整 Transformer，并在大规模翻译数据集上训练。

本项目保留了核心思想：

- encoder-decoder Transformer；
- multi-head attention；
- positional encoding；
- masking；
- token embedding；
- cross-entropy loss；
- BLEU 评估。

但为了课程作业和本地训练，项目做了简化：

- 使用较小的 IWSLT2017 数据集；
- 使用较小模型；
- 使用 PyTorch 自带 `nn.Transformer`；
- 使用 greedy decoding；
- 没有实现完整 beam search；
- 没有使用大规模训练技巧。

### 7.14 Transformer 在本项目代码中的对应关系

论文里的 Transformer 概念，在本项目中主要对应这些代码：

```text
Embedding                 -> self.src_emb / self.tgt_emb
Positional Encoding       -> PositionalEncoding
Encoder-Decoder Transformer -> nn.Transformer
Multi-Head Attention      -> nn.Transformer 内部实现
Mask                      -> tgt_mask
Padding Mask              -> src_key_padding_mask / tgt_key_padding_mask
Output Projection         -> self.out
Cross Entropy Loss        -> nn.CrossEntropyLoss
```

也就是说，本项目没有手写每一个 attention 细节，而是使用 PyTorch 的 `nn.Transformer` 来承载核心结构。这样代码更短，适合课程项目；同时仍然保留了 Transformer 翻译模型的主要组成部分。

### 7.15 训练阶段和翻译阶段的区别

训练阶段和真正翻译阶段是不一样的。

训练时，标准英文答案已经存在，所以模型可以使用 teacher forcing：

```text
给模型中文句子 + 正确英文前缀
让模型预测下一个英文 token
```

例如：

```text
输入中文：我喜欢机器翻译
decoder 输入：BOS I like machine
模型目标：I like machine translation
```

这样训练速度比较快，也比较稳定。

真正翻译时，没有标准英文答案。模型只能从 `BOS` 开始，自己一个 token 一个 token 生成：

```text
BOS -> I -> like -> machine -> translation -> EOS
```

当前代码使用 greedy decoding，每一步都选概率最高的 token。它简单、容易解释，但有时会导致重复、漏译或局部最优。例如模型可能很早选错一个词，后面就会被这个错误影响。

更好的方法是 beam search。Beam search 会同时保留多个候选句子，而不是每一步只保留一个最优 token。这样通常能提高翻译质量，但代码会更复杂，所以本项目先保留 greedy decoding。

### 7.16 为什么小模型效果有限

当前模型参数比较保守：

```python
d_model = 256
nhead = 4
num_layers = 3
```

这个配置适合在 RTX 4060 这类消费级显卡上完成训练，但它比论文中的大模型小很多。

机器翻译，尤其是中英翻译，需要学习大量词汇、短语、语序变化和上下文语义。小模型可以学到基本对应关系，但在长句、复杂句、专业表达上容易出错。

因此当前结果中会出现：

- 简单短句翻译较接近；
- 长句容易漏译；
- 有时生成重复片段；
- 有时语义发生偏移。

这不是代码流程错误，而是小模型、有限训练轮数、greedy decoding 共同导致的正常现象。

## 8. 实验结果如何理解

当前结果：

```text
BLEU = 6.39
```

这个分数不算高。

原因包括：

- 模型较小；
- 训练轮数有限；
- 使用 greedy decoding；
- 没有使用预训练模型；
- 中英翻译本身难度较高；
- IWSLT 句子较长，包含演讲语境。

但从样例可以看到，模型不是完全随机输出。它能学到一些基本对应关系。

例如：

```text
Source: 这个真是很神奇
Reference: And it's pretty amazing.
Prediction: That's amazing.
```

这个例子翻译得比较接近。

也有明显错误：

```text
Source: 我喜欢机器翻译。
Prediction: I love machines.
```

模型翻出了“我喜欢”和“机器”，但漏掉了“翻译”。

因此可以总结为：

```text
模型成功跑通了 Transformer 中英翻译流程，能够学习到部分语义对应关系；
但由于模型规模和训练条件有限，复杂句子上仍存在漏译、重复和语义偏移。
```

## 9. 如果想提高效果

可以从以下方向改进：

### 9.1 增加训练轮数

把：

```python
epochs = 20
```

改成：

```python
epochs = 40
```

训练时间会变长，但效果可能提升。

### 9.2 增大模型

当前模型：

```python
d_model = 256
num_layers = 3
nhead = 4
```

可以尝试：

```python
d_model = 512
num_layers = 4
nhead = 8
```

但显存占用和训练时间都会增加。

### 9.3 使用 beam search

当前是 greedy decoding，每一步只选择概率最高的词。

Beam search 会同时保留多个候选翻译，通常翻译质量更好。

### 9.4 调整学习率

如果 loss 下降不稳定，可以降低学习率，例如：

```python
learning_rate = 0.0003
```

### 9.5 使用更大数据集

如果算力充足，可以尝试 WMT 数据集。但对普通课程项目来说，IWSLT2017 已经比较合适。

## 10. 常见问题

### 10.1 为什么 preprocess 只需要跑一次

因为它的作用是下载和整理数据。

只要 `data/` 目录里已经有：

```text
train.zh
train.en
valid.zh
valid.en
test.zh
test.en
spm.model
```

后面就可以直接训练。

### 10.2 为什么 evaluate 很久没有输出

因为 evaluate 会逐句翻译测试集。

测试集有：

```text
8549
```

条句子。每一句都要 autoregressive 生成英文，所以会比较慢。

### 10.3 PyTorch nested tensor warning 是错误吗

不是错误。

运行时可能看到：

```text
UserWarning: The PyTorch API of nested tensors is in prototype stage...
```

这是 PyTorch 内部优化路径的提示，不影响结果。

### 10.4 为什么不直接用 ChatGPT 翻译

本项目的目的不是获得最好翻译结果，而是理解机器翻译模型如何从数据中学习。

ChatGPT 是大规模预训练模型，而本项目是从零训练一个小型 Transformer。

课程项目重点是：

```text
数据 -> 模型 -> 训练 -> 评估
```

这个完整过程。

## 11. 项目总结

本项目完成了一个完整的中英机器翻译实验：

- 使用 IWSLT2017 中英平行语料；
- 使用 SentencePiece 训练子词分词器；
- 使用 PyTorch 搭建 encoder-decoder Transformer；
- 使用交叉熵损失训练模型；
- 保存模型和 loss 曲线；
- 使用 BLEU 评价翻译效果；
- 输出翻译样例分析模型表现。

一句话总结：

```text
这是一个面向课程作业的简化版 Transformer 中英翻译复现实验。
```



