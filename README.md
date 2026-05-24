# Multi-Head-Attention

### （1）题目意思与生僻概念解释

这段代码实现的是深度学习Transformer架构中的核心模块——**多头注意力机制（Multi-Head Attention）**。简单来说，它的功能是从一段输入数据中，计算出每个元素与其他所有元素的关联程度，从而让模型知道在处理当前元素时，应该把“注意力”集中在哪些相关的上下文信息上。

**生僻概念解释：**

* **Q、K、V（Query, Key, Value）**：可以类比为数据库的检索过程。Q（查询词）代表当前需要处理的元素所发出的搜索请求；K（键）代表所有元素的特征标签；V（值）代表所有元素的实际内容。算法通过计算 Q 和 K 的匹配度，来决定最终提取多少对应的 V。
* **多头（Multi-Head）**：如果只用一套 Q、K、V，模型可能只能学到一种维度的相关性。多头机制将原始的高维数据切分成多个低维度的“头”，每个头独立计算注意力，就像安排多个人从不同的角度（如语法、情感、逻辑）去阅读同一段文字，最后再把这些视角的结论合并起来。
* **d_model 与 d_k**：`d_model` 是输入数据的总维度大小（例如512）。`d_k` 是将其切分给多个头之后，每个头负责的特征维度（例如8个头，`d_k` 就是 512/8 = 64）。
* **Mask（掩码）**：在某些场景下（如生成文本时，不能看到未来的词；或者输入长度不一，需要填充无意义的0），需要用掩码把特定位置遮挡住。代码中通过赋予极小的值（$-1 \times 10^9$），使得对应位置在计算概率时趋近于0，从而被模型忽略。

---

### （2）代码解题思路解析

从代码结构进行逆推，该算法的执行逻辑分为两大阶段：**初始化参数阶段**与**前向传播计算阶段**。

1. **参数构建层（`__init__`）**：首先确定特征总维度和拆分的头数，计算出单个头的维度。接着声明四个至关重要的可学习网络层：三个用于将原始输入分别映射为特定 Q、K、V 空间的线性变换层，以及一个用于将最终多头拼接结果整合输出的线性变换层。
2. **空间变换与切分**：在执行前向传播（`forward`）时，输入数据先经过三个独立的线性层转换为 Q、K、V 张量。随后，代码利用维度重塑（`view`）和维度交换（`transpose`）操作，将这些张量从“完整的单一样本”切割成“并行的多个头”，为接下来的并发计算做准备。
3. **核心注意力计算（`attention`）**：这是整个机制的心脏。核心数学公式为 $\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$。
* 先计算 Q 与 K 的转置的内积，得到原始的相似度得分矩阵。
* 除以缩放因子 $\sqrt{d_k}$，防止内积结果过大导致 Softmax 梯度消失。
* 根据掩码（如果有），将不需要关注的位置分值拉至极低。
* 经过 Softmax 激活函数，将得分转化为总和为1的概率权重矩阵。
* 将该权重矩阵与 V 相乘，得出加权后的特征表示。


4. **特征重组**：各个头计算出的结果分散在不同的维度中。此时再次使用维度交换和重塑操作，将它们在特征维度上拼接起来，恢复为最初的序列形态。
5. **最终融合**：拼接后的数据虽然维度对齐了，但各个头之间的数据还没有交互，最后通过一个输出线性层（`self.out`）进行信息融合，输出最终的处理结果。

---

### （3）带有详细注释的代码

```python
from math import sqrt                 # 01. 从数学库导入平方根计算函数
import torch                          # 02. 导入PyTorch张量运算基础库
import torch.nn as nn                 # 03. 导入PyTorch神经网络核心模块

class MultiHeadAttention(nn.Module):  # 04. 定义多头注意力类，继承自神经网络基础类
    def __init__(self, heads, d_model, dropout=0.1): # 05. 初始化方法，接收头数、特征总维度和失活率
        super(MultiHeadAttention, self).__init__()   # 06. 调用父类的初始化方法
        self.d_model = d_model        # 07. 将输入的特征总维度保存为类的属性
        self.d_k = d_model // heads   # 08. 计算每个注意力头的特征维度（总维度整除头数）
        self.h = heads                # 09. 将注意力头的总数量保存为类的属性
        
        self.q_linear = nn.Linear(d_model, d_model)  # 10. 定义用于生成Query的全连接映射层
        self.v_linear = nn.Linear(d_model, d_model)  # 11. 定义用于生成Value的全连接映射层
        self.k_linear = nn.Linear(d_model, d_model)  # 12. 定义用于生成Key的全连接映射层
        
        self.dropout = nn.Dropout(dropout)           # 13. 定义Dropout层，训练时随机丢弃以防过拟合
        self.out = nn.Linear(d_model, d_model)       # 14. 定义输出的全连接映射层，用于整合多头拼接特征

    def attention(self, q, k, v, d_k, mask=None, dropout=None): # 15. 定义内部核心注意力计算方法
        # 16. 计算注意力得分：Q乘上转置后的K，并除以缩放因子。结果形状: (batch, heads, seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_k)
        
        if mask is not None:                         # 17. 判断是否传入了掩码张量
            mask = mask.unsqueeze(1)                 # 18. 增加一个维度，使掩码能与多头形状广播对齐
            scores = scores.masked_fill(mask == 0, -1e9) # 19. 将掩码中值为0的位置在得分矩阵中替换为极小值
            
        scores = torch.softmax(scores, dim=-1)       # 20. 对最后一个维度执行Softmax归一化，生成概率分布权重
        
        if dropout is not None:                      # 21. 判断是否传入了随机失活(dropout)模块
            scores = dropout(scores)                 # 22. 对生成的注意力权重应用随机丢弃
            
        output = torch.matmul(scores, v)             # 23. 注意力权重矩阵与Value矩阵相乘进行特征加权求和
        return output                                # 24. 返回单次注意力聚合计算完毕的张量

    def forward(self, q, k, v, mask=None):           # 25. 定义前向传播函数
        bs = q.size(0)                               # 26. 获取输入张量的批次大小(batch_size)
        
        k = self.k_linear(k)                         # 27. 将输入映射为Key矩阵，形状: (bs, seq_len, d_model)
        q = self.q_linear(q)                         # 28. 将输入映射为Query矩阵，形状: (bs, seq_len, d_model)
        v = self.v_linear(v)                         # 29. 将输入映射为Value矩阵，形状: (bs, seq_len, d_model)
        
        # 30. 重构Key维度：利用view切分多头，再用transpose交换维度变为 (bs, heads, seq_len, d_k)
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        # 31. 重构Query维度：利用view切分多头，再用transpose交换维度变为 (bs, heads, seq_len, d_k)
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        # 32. 重构Value维度：利用view切分多头，再用transpose交换维度变为 (bs, heads, seq_len, d_k)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # 33. 调用内部attention方法计算多头注意力结果，返回形状为 (bs, heads, seq_len, d_k)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 34. 维度还原与拼接：换回原位并确保存储连续，最后展平合并回总维度 (bs, seq_len, d_model)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)                    # 35. 将合并后的特征输入最终线性映射层
        return output                                # 36. 返回最终多头注意力处理完毕的输出张量

```

---

### （4）逐行详细中文解释

* **第1-4行**：导入PyTorch核心库、神经网络模块、函数式操作模块以及Python内置的数学库。
* **第6行**：定义名为 `MultiHeadAttention` 的类，继承自 PyTorch 的基础网络模块 `nn.Module`。
* **第7行**：定义类的初始化方法，接收头数 `heads`，模型维度 `d_model`，以及默认值为0.1的丢弃率 `dropout`。
* **第8行**：调用父类的初始化方法，确保网络模块能够被 PyTorch 正确追踪。
* **第10行**：将传入的模型特征总维度作为类属性保存。
* **第12行**：利用整除计算出拆分后单个注意力头的特征维度 `d_k`。
* **第14行**：将注意力头的数量保存到类属性 `h` 中。
* **第17-19行**：实例化三个全连接层，输入和输出维度均为 `d_model`，分别负责生成当前批次的查询矩阵 Q、键矩阵 K、值矩阵 V。
* **第22行**：实例化 Dropout 层，用于后续在注意力权重上进行正则化，抑制过拟合现象。
* **第25行**：实例化输出全连接层，用于在多头结果合并后进行最后一次特征整合。
* **第27行**：定义内部使用的注意力计算方法，接收切分好的 q, k, v、单个头维度、掩码及丢弃网络。
* **第32行**：执行矩阵乘法计算 q 与 k 转置的内积，并立即除以 $\sqrt{d_k}$ 进行数值缩放，结果存入 `scores`。
* **第35行**：判断是否传入了用于遮挡信息的掩码张量。
* **第36行**：若有掩码，则在索引为1的维度（即 `heads` 的位置）增加一维，以匹配多头计算的张量形状。
* **第37行**：利用 `masked_fill` 方法，找到掩码中数值为0的对应位置，将其在 `scores` 中的数值强行替换为负十亿（$-1 \times 10^9$）。
* **第40行**：沿着倒数第一个维度应用 Softmax 激活函数，使所有得分转换为区间在 0 到 1 之间、且每行加和为 1 的概率。
* **第43-44行**：如果定义了丢弃网络，则在这些概率权重上随机将部分值归零并按比例放大其余值。
* **第49行**：将概率矩阵 `scores` 与值矩阵 `v` 进行矩阵乘法运算，实现对特征向量的加权求和。
* **第50行**：返回加权求和后的结果，此结果为当前各个头独立计算所得。
* **第52行**：定义网络的前向传播主逻辑方法，这是数据流入类时的默认执行入口。
* **第53行**：提取输入张量 `q` 的第一个维度的尺寸，这代表当前处理的数据批次大小（Batch Size）。
* **第58-60行**：让输入的原始序列依次通过之前初始化的三个线性变换层，获得全局的 Q、K、V 表示。
* **第65-67行**：这是切分多头的关键步骤。`view(bs, -1, self.h, self.d_k)` 将最后一个维度 `d_model` 切分为 `(heads, d_k)`；随后的 `.transpose(1, 2)` 将序列长度维度和头维度位置互换，使得各个头的计算能在内存空间中独立且并行地进行。
* **第70行**：调用自身类中定义的 `attention` 方法，输入多头形状的张量进行核心运算。
* **第75行**：首先 `.transpose(1, 2)` 将分离的头维度重新换回后面，接着 `.contiguous()` 强制重新分配连续的底层内存，最后 `.view(bs, -1, self.d_model)` 将多头重叠合并回单一的特征维度 `d_model`。
* **第78行**：将重组归一后的张量传入输出映射层。
* **第79行**：将最终结果返回至外部调用者。

---

### （5）具体数值算例与追踪过程表格

假设采用一个极简的环境用于推演：

* 批次大小 `batch_size = 1`
* 序列长度 `seq_len = 2` （即包含两个词向量输入）
* 总维度 `d_model = 4`，注意力头数 `heads = 2` （因此 `d_k = 2`）。
* 为了方便展示计算核心，假设 Q, K, V 对应的线性映射网络层的权重均为单位矩阵（即输入原样输出）。

输入序列 `X` 初始状态为：
`[[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]`

以下是关键变量演化表格：

| 步骤 | 变量/张量形状及数值变化 | 对应代码行 | 对应代码 |
| --- | --- | --- | --- |
| **初始** | **X 形状**: `(1, 2, 4)` <br>

<br> **值**: `[[[1, 0, 1, 0], [0, 1, 0, 1]]]` | 行 58-60 | `k = self.k_linear(k)` <br>

<br> `q = ...` <br>

<br> `v = ...` |
| **重塑拆分** | 将 `d_model` (4) 拆为 `heads` (2) × `d_k` (2)。<br>

<br>先 `view` 成 `(1, 2, 2, 2)`。 <br>

<br> $\rightarrow$ `[[[[1,0], [1,0]], [[0,1], [0,1]]]]` | 行 65-67 | `k = k.view(...).transpose(...)` |
| **转置换位** | 执行 `transpose(1, 2)` 交换 `seq_len` 和 `heads`。<br>

<br>**q, k, v 形状**: `(1, 2, 2, 2)`。 <br>

<br>头1: `[[1,0], [0,1]]` <br>

<br>头2: `[[1,0], [0,1]]` | 行 65-67 | `k = k.view(...).transpose(...)` |
| **内积计算** | 第一个头: $\text{Q}_1 \times \text{K}_1^T$ <br>

<br>`[[1,0], [0,1]]` $\times$ `[[1,0], [0,1]]` $\rightarrow$ `[[1,0], [0,1]]` <br>

<br>同理计算第二个头。 | 行 32 | `scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)` |
| **缩放得分** | 每项除以 $\sqrt{2} \approx 1.414$。<br>

<br>**得分矩阵值**: <br>

<br> $\rightarrow$ `[[0.707, 0], [0, 0.707]]` | 行 32 | `... / math.sqrt(d_k)` |
| **Softmax** | 针对每一行计算 Softmax 转化为概率。 <br>

<br>例如行1: $e^{0.707} / (e^{0.707} + e^{0}) \approx 2.028 / 3.028 \approx 0.67$<br>

<br>$e^{0} / 3.028 \approx 0.33$<br>

<br> $\rightarrow$ `[[0.67, 0.33], [0.33, 0.67]]` | 行 40 | `scores = F.softmax(scores, dim=-1)` |
| **加权求和** | `scores` $\times$ `v`。对应头1计算：<br>

<br>`[[0.67, 0.33], [0.33, 0.67]]` $\times$ `[[1,0], [0,1]]`<br>

<br> $\rightarrow$ `[[0.67, 0.33], [0.33, 0.67]]` | 行 49 | `output = torch.matmul(scores, v)` |
| **恢复顺序** | 执行 `transpose(1, 2)` 将形状从 `(1, 2, 2, 2)` 换回 `(1, 2, 2, 2)` （因数据特殊，表象未变，但内存结构重排了顺序，保证词维度相邻）。 | 行 75 | `concat = scores.transpose(1, 2).contiguous()...` |
| **拼接合并** | 执行 `.view(bs, -1, self.d_model)` 将头维度和特征维度压平。<br>

<br>**concat 形状**: `(1, 2, 4)` <br>

<br> $\rightarrow$ `[[[0.67, 0.33, 0.67, 0.33], [0.33, 0.67, 0.33, 0.67]]]` | 行 75 | `... .view(bs, -1, self.d_model)` |
| **最终输出** | 送入最后的线性层整合（假设权重仍为单位阵）。<br>

<br>**结果形状**: `(1, 2, 4)` <br>

<br>计算结束。 | 行 78 | `output = self.out(concat)` |
