---
layout: post
title: "矢量语义(vector semantics)笔记"
summary: 从TF-IDF到word2vec你所需要知道的一切
featured-img: shane-rounce-205187
---

斯坦福经典NLP教材[Speech and Language Processing-Vector Semantics](https://web.stanford.edu/~jurafsky/slp3/6.pdf)学习笔记。

我们该如何表示一个单词的意思呢？你可能会想到其中的一种，用一个向量来表示一个单词！没错，这个章节就是讲单词的表示。

## 文档和向量

如果用向量来表示一个文档，该怎么表示呢？

假设现在有四个文档，我们统计各个单词在文档中出现的次数，可以得到一张表格：

|-|As You Like It|Twelfth Night|Julius Caesar|Henry V|
|:-----|:----|:----|:----|:----|
|battle|1    |0    |7    |13   |
|good  |114  |80   |62   |89   |
|fool  |36   |58   |1    |4    |
|wit   |20   |15   |2    |3    |

当然实际上文档的单词数量远不止这几个。

上面表中，有4个单词，所以每一个文档可以表示成一个由单词频率组成的向量：

```bash
As You Like It ------> [ 1,114,36,20]
 Twelfth Night ------> [ 0, 80,58,15]
 Julius Caesar ------> [ 7, 62, 1, 2]
       Henry V ------> [13, 89, 4, 3]
```

如果单词有很多个，假设是`N`，那么每个文档就可以表示成一个`N`维的向量。可见，这样的向量表示是**稀疏的(sparse)**。

## 单词和向量

除了文档可以表示成一个向量，单词也可以。

和文档类似，我们可以统计出一张表格，但是不同的是，我们不是统计单词的个数，而是统计两个单词出现在一起的频数。看一张表格你就知道了：

|-|aardvark|...|computer|data|pinch|result|sugar|
|:------|:---|:---|:---|:---|:---|:---|:---|
|apricot|0   |... |0   |0   |1   |0   |1   |
|pineapple|0 |... |0   |0   |1   |0   |1   |
|digital|0   |... |2   |1   |0   |1   |0   |
|information|0|...|1   |6   |0   |4   |0   |
|...    ||||||||

这个表格是一个$V\times V$的表格，每个数字表示当前列的单词出现在当前行单词后面的次数，这就构成了上下文，所以这个表格其实就是一个上下文矩阵，其中`V`就是总的词典的大小，也就是单词的数量。

我们取出每一行，就可以得到一个单词的向量表示，例如：

```bash
digital ------> [ 0,..., 2, 1, 0, 1, 0]
```
同样的，这样的表示也是**稀疏的**。

## Cosine计算相似度

现在我们已经有文档或者单词的向量表示了，那么该如何计算它们之间的相似度呢？一个很常见的方法就是**余弦相似度(Cosine similarity)**。

学过高中数学就知道，两个向量的**点积(dot-product)**或者**内积(inner product)**可以由以下公式计算：

$$\text{dot-produtc}(\overrightarrow{v},\overrightarrow{w}) = \sum_{i=1}^Nv_iw_i=v_1w_1+v_2w_2+\dots+v_Nw_N$$

而**向量的模(vector length)**为：

$$\vert\overrightarrow{v}\vert = \sqrt{\sum_{i=1}^Nv_i^2}$$

又:

$$\overrightarrow{a}\cdot\overrightarrow{b} = \vert{\overrightarrow{a}}\vert \vert{\overrightarrow{b}}\vert \cos\theta$$

即：

$$\cos\theta = \frac{\overrightarrow{a}\cdot\overrightarrow{b}}{\vert{\overrightarrow{a}}\vert \vert{\overrightarrow{b}}\vert}$$

所以，我们可以计算$\overrightarrow{v}$和$\overrightarrow{w}$的余弦值：

$$\cos(\overrightarrow{v},\overrightarrow{w}) = \frac{\overrightarrow{v}\cdot\overrightarrow{w}}{\vert{\overrightarrow{v}}\vert \vert{\overrightarrow{w}}\vert} = \frac{\sum_{i=1}^Nv_iw_i}{\sqrt{\sum_{i=1}^Nv_i^2}\sqrt{\sum_{i=1}^Nw_i^2}}$$

所以，两个向量的余弦值越大，它们越相似。

## TF-IDF

接下来就要介绍TF-IDF了，首先解释一下这个词：

```bash
TF-IDF = Term Frequency - Inverse Document Frequency
```

理解了名称，你就理解了一半！

那么什么是`term-frequency`呢？**term-frequency**就是单词在文档中出现的次数。

$$\text{tf}_{t,d} = 1+\log_{10}{\text{count}(t,d)} \quad\text{if }\text{count}(t,d) > 0 \quad\text{else } 0$$

那么什么是**IDF**呢？首先我们弄清楚**DF(document frequency)**。

$\text{df}_t$表示单词在所有文档(document)中出现的次数！

那么，**IDF**就是：

$$\text{idf}_t = \frac{N}{\text{df}_t}$$

其中，`N`就是一个集合(collection)中的documents数量。

为了避免数值过大，通常会取对数：

$$\text{idf}_t = \log_{10}(\frac{N}{\text{df}_t})$$

至此，我们可以计算这个单词$t$的`tf-idf`权值：

$$w_{t,d} = \text{tf}_{t,d}\times\text{idf}_t$$

此时，我们的第一个表格，就变成了：

|-|As You Like It|Twelfth Night|Julius Caesar|Henry V|
|:-----|:----|:----|:----|:----|
|battle|0.074|0    |0.22 |0.28 |
|good  |0    |0    |0    |0    |
|fool  |0.019|0.021|0.0036|0.0083|
|wit   |0.049|0.044|0.018 |0.022|

到目前为止，上面的所有向量表示都是**稀疏的**，接下来要介绍一种**稠密的(dense)**的向量表示——**word2vec**！

## word2vec

这个大家应该很熟悉了，应该算是NLP领域的标配了。


