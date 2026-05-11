# Multi-Head-Attention

### （1）题目含义与生僻概念白话文解释

**题目含义**
这段代码实现了“多头注意力机制（Multi-Head Attention）”。这是在自注意力机制基础上的升级版本。相比于只用一种方式去观察数据，多头注意力机制将输入特征拆分成多个不同的子集（即多个“头”），让这些“头”独立地去计算注意力权重。这样能够让模型同时从多个不同的角度去捕捉输入序列中的信息。

**生僻概念解释**

* **多头（Multi-Head）**：相当于请了多位不同领域的专家，每个人只负责分析事物的一个特定方面，最后把所有专家的意见汇总。
* **掩码（Mask）**：可以理解为“遮罩层”。在处理长短不一的序列或需要防止提前看到后续信息时，使用掩码将特定位置的注意力得分强行设为极小值，使其在计算权重时趋近于0。
* **连续化（Contiguous）**：在底层内存中，张量经过维度交换操作后，其内存地址可能不再连续。调用此方法可强制在内存中重新分配一块连续空间，以满足后续改变形状（view）操作的底层计算要求。

---

### （2）代码视角的解题思路

这段代码的核心逻辑在于特征切分与并行处理。首先通过线性层将输入映射到特征空间，随后利用维度重塑操作将高维特征拆分为多个独立的头。各个头并行执行标准的缩放点积注意力计算。得到多组注意力结果后，再将其重新拼接并经过一次线性映射还原至原始维度，从而综合出包含多种上下文关系的最终特征表示。

---

### （3）带有详细注释的代码框

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model                    # 保存模型整体特征维度
        self.d_k = d_model // heads               # 计算单个头分配到的特征维度
        self.h = heads                            # 保存总头数
        
        self.q_linear = nn.Linear(d_model, d_model) # 定义Q映射层
        self.v_linear = nn.Linear(d_model, d_model) # 定义V映射层
        self.k_linear = nn.Linear(d_model, d_model) # 定义K映射层
        
        self.dropout = nn.Dropout(dropout)          # 定义随机失活层
        self.out = nn.Linear(d_model, d_model)      # 定义输出融合层

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # 计算缩放点积
        
        if mask is not None:
            mask = mask.unsqueeze(1)                # 维度对齐
            scores = scores.masked_fill(mask == 0, -1e9) # 应用掩码
            
        scores = F.softmax(scores, dim=-1)          # 转换为概率分布
        
        if dropout is not None:
            scores = dropout(scores)                # 应用失活
            
        output = torch.matmul(scores, v)            # 加权聚合
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)                              # 提取批量大小
        
        k = self.k_linear(k)                        # 生成K张量
        q = self.q_linear(q)                        # 生成Q张量
        v = self.v_linear(v)                        # 生成V张量
        
        # 拆分多头并调整维度顺序，以支持批量并行运算
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # 调用核心注意力计算函数
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 还原维度顺序并拼接所有头的信息
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)                   # 最终线性映射
        return output

```

---

### （4）每一行代码详细的中文解释

* **1-4行**：导入PyTorch基础库、神经网络模块、激活函数库以及Python内置数学库。
* **6-7行**：声明多头注意力类，继承神经网络基础模块，并初始化参数：头数、特征维度、失活率。
* **8行**：调用父类初始化逻辑。
* **9行**：将总特征维度数值持久化保存。
* **10行**：使用总维度整除头数，得出每个头需要处理的子特征维度（`d_k`）。
* **11行**：持久化保存头的数量。
* **13-15行**：实例化三个全连接线性层，分别用于将输入数据转换为多头机制所需的Q、K、V内部表示。
* **17行**：实例化Dropout层，用于在训练期间随机切断部分连接，降低过拟合风险。
* **19行**：实例化一个线性层，负责把多头拼接后的张量再次混合映射，输出最终结果。
* **21行**：定义内部使用的核心注意力计算函数。
* **22行**：对最后两个维度执行矩阵乘法，并除以维度平方根，完成点积缩放计算。
* **24-26行**：判断是否传入了掩码张量。如果有，则增加一个维度以对齐多头形状，并将掩码值为0对应位置的得分替换为极小的负数（`-1e9`）。
* **28行**：在最后一个维度上执行Softmax，将原始得分转化为和为1的相对权重。
* **30-31行**：判断是否传入Dropout操作，如有则对权重分布进行随机失活处理。
* **33-34行**：将计算好的权重张量与值张量进行矩阵相乘，得出聚合后的局部特征，并返回。
* **36-37行**：定义类的前向传播函数。提取输入数据第一维度的数值作为批量样本数量（`bs`）。
* **39-41行**：将外界传入的原始Q、K、V数据分别通过相应的线性层，得到映射后的新特征空间表示。
* **44-46行**：利用 `view` 将特征维度拆解为“头数 × 单头维度”，随后用 `transpose` 调换“序列长度”与“头数”的维度位置，使得张量形状变为 `(batch, heads, seq_len, d_k)`。
* **49行**：将变形后的数据送入上述定义的 `attention` 函数中，完成各个独立头的并行计算。
* **53行**：先将维度位置换回 `(batch, seq_len, heads, d_k)`，调用 `contiguous` 确保存储连续后，利用 `view` 强制将后两个维度合并回原始的 `d_model`，完成多头信息拼接。
* **56-57行**：将拼接完成的高维张量经过最后一道线性映射层，返回处理完毕的输出。

---

### （5）具体数值算例与关键变量变化过程表

**【前置设定】**
批量大小 `bs=1`，序列长度 `seq=2`，总特征维度 `d_model=4`，头数 `heads=2`。
在此设置下，单头维度 `d_k = 4 // 2 = 2`。
假设所有的线性层初始权重均为单位矩阵，且不考虑Dropout和Mask。
初始输入特征张量经过映射后 $Q=K=V = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \end{bmatrix}$，形状为 `(1, 2, 4)`。

**【执行过程记录表】**

| 步骤序号 | 变量名称与形状 (Shape) | 关键变量数值变化示意 (Markdown表示) | 对应代码行号 | 代码语句内容 |
| --- | --- | --- | --- | --- |
| **步骤1** | `q`<br>

<br> `(1, 2, 4)` | $$Q = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \end{bmatrix}$$

<br> <br>

<br> *初始映射后的矩阵特征* | 第40行 | `q = self.q_linear(q)` |
| **步骤2** | `q.view(...)`<br>

<br> `(1, 2, 2, 2)` | 依据头数进行特征切割拆分：<br>

<br>头1：$\begin{bmatrix} 1 & 2 \\ 5 & 6 \end{bmatrix}$，头2：$\begin{bmatrix} 3 & 4 \\ 7 & 8 \end{bmatrix}$ | 第45行 | `q = q.view(bs, -1, self.h, self.d_k)...` |
| **步骤3** | `q` (transpose)<br>

<br> `(1, 2, 2, 2)` | 交换维度以适应批处理计算：<br>

<br>形状转变为 `(bs, heads, seq, d_k)` | 第45行 | `...transpose(1, 2)` |
| **步骤4** | `scores`<br>

<br> `(1, 2, 2, 2)` | 分别在两个头内并行计算自注意力加权后的结果。<br>

<br>输出的 `scores` 形状与切分后的形状一致。 | 第49行 | `scores = self.attention(...)` |
| **步骤5** | `scores` (转置回)<br>

<br> `(1, 2, 2, 2)` | 将维度重整回 `(bs, seq, heads, d_k)`，准备拼接。 | 第53行 | `concat = scores.transpose(1, 2)...` |
| **步骤6** | `concat`<br>

<br> `(1, 2, 4)` | 合并各个头的结果，还原回总特征维度：<br>

<br> $\rightarrow \begin{bmatrix} f_{1,1} & f_{1,2} & f_{1,3} & f_{1,4} \\ f_{2,1} & f_{2,2} & f_{2,3} & f_{2,4} \end{bmatrix}$ | 第53行 | `...view(bs, -1, self.d_model)` |
| **步骤7** | `output`<br>

<br> `(1, 2, 4)` | 通过输出层矩阵运算，得到多头处理终值。 | 第56行 | `output = self.out(concat)` |
