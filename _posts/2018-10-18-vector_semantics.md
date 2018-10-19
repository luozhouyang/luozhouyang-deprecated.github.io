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

$\text{df}_t$表示出现过这个单词的文档(document)的个数！

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

## Word2Vec

这个大家应该很熟悉了，应该算是NLP领域的标配了。

我之前写过一篇word2vec的笔记[自己动手实现word2vec(skip-gram模型)](https://juejin.im/post/5b986f296fb9a05d11176a15)，但是其实还是很粗糙。Tensorflow也有一个教程[Vector Representations of Words](https://www.tensorflow.org/tutorials/representation/word2vec)，但是如果你没有一点基础的话，也还是有些概念难以理解。所以相对完整地理解word2vec，你需要结合多方面的资料。这个笔记在介绍斯坦福教材的同时，也会引入其他文章，做一些比较和思考，希望这个笔记能够给你带来相对全面的理解。

### word embedding

首先我们解释一下**词嵌入(word embedding)**的概念。本小节之前的所有向量表示都是稀疏的，通常都是一个高维的向量，向量里面的元素大部分都是0。那么**embedding**有什么不一样的呢？

**Embedding同样也是用一个向量来表示一个词，但是它是使用一个较低的维度，稠密地表示**。

如果使用之前的稀疏表示，你可能会这样表示`hello`这个词语：

$$\text{hello} \longrightarrow \quad\underbrace{[0, 0, 0, 1, 2, 0,\dots, 0]}_{N个数}$$

那么，使用**嵌入**表示之后会是什么样子呢：

$$\text{hello} \longrightarrow \quad\underbrace{[0.012,0.025,0.001,0.078,0.056,0.077,\dots,0.022]}_{n个数，一般是100到500左右}$$

其中的差异一眼就看出来了。所以很明显，word embedding有好处：

* 不会造成维度爆炸，因为维度是我们自己设置的，通常比较小
* 向量是稠密的，不需要稀疏向量所采用的各种优化算法来提升计算效率

词嵌入理解了，那么什么是word2vec呢？其实就是**把单词表示成固定维度的稠密的向量**！说起来简单，但是也有很多小技巧的。

### 数据模型

假设我们有一个很大的文本语料，我们需要用这个语料来训练出单词的向量表示。那么该怎么训练呢？

当然你可能会想到基于计数的方式，就像前面几个小节一样，我们不说这个。

word2vec有两种常用的数据准备方式：

* CBOW，用前后词(context words)预测目标词(target word)
* skip-gram，用目标词(target word)预测前后词(context word)

使用tensorflow里面的例子：

```bash
the quick brown fox jumped over the lazy dog
```

举个例子，假设我们的**窗口大小(window size)**是`2`，**目标词**选择`fox`。

如果是**skip-gram**模型，我们会这样准备数据：

```bash
(fox, quick)
(fox, brown)
(fox, jumped)
(fox, over)
```

也就是一个目标词，我们可以构造出`window_size`个训练数据对。

如果是**CBOW**模型，我们会这样准备数据：

```bash
([quick brown jumped over], fox)
```

看出其中的差异了吧？

总之，**skip-gram**和**CBOW**就是两个相反的数据模型。[Learning Word Embedding](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)有两张图可以分别表示两种模型的输入方式：

<img src="http://blog.stupidme.me/wp-content/uploads/2018/10/word2vec-skip-gram.png" style="width:500px">

*skip-gram模型*

<img src="http://blog.stupidme.me/wp-content/uploads/2018/10/word2vec-cbow.png" style="width:500px">

*CBOW模型*

数据模型应该清楚了。

与之前不同的是，**word2vec并不关心相邻单词之前一起出现的频数，而是仅仅关心，这个单词是不是属于另一个单词的上下文(context)**!也就是说，word2vec不关系根据这个词预测出的下一个词语是什么，而是只关心这两个词语之间是不是有上下文关系。

于是，word2vec需要的仅仅是一个**二分类器**：“这个单词是另一个单词的上下文单词吗？”

所以，要训练一个word2vec模型，我们其实是在训练一个二分类器。而二分类器，你肯定很容易就想到了**Logistic Regression**！关于逻辑回归，可以看我的另一篇笔记[Logistic Regression](https://luozhouyang.github.io/logistic_regression/)。

实际情况，skip-gram用的比较多，因为有一个说法，CBOW模型在小的数据集上面表现不错，在大的数据集里，skip-gram表现更好。

### 神经语言模型

这里需要说明进一步说明一下。Tensorflow里面有关于**神经概率语言模型(nerual probability language model)**的描述。

传统的神经概率语言模型的训练通常是用**最大似然(maximum likelihood)**法则来最大化下一个词的softmax概率，基于前面的词，也就是：

$$P(w_t\vert h) = \text{softmax}(\text{score}(w_t,h)) = \frac{\exp{\text{score}(w_t,h)}}{\sum_{\text{w' in V}}\exp{\text{score}(w',h)}}$$

其中，$\text{score}(w_t,h)$其实就是$w_t$和$h$的**点积(dot-production)**。

那么这样训练模型的目标就是，最大化**对数似然概率(log likelihood)**：

$$J_{\text{ML}} = \log{P(w_t\vert h)} = \text{score}(w_t,h) - \log(\sum_{\text{w' in V}}\exp{\text{score}(w',h)})$$

那么这样会有什么问题吗？**计算量太大了**，因为在每一个训练步里，需要对词典里的每一个词，使用softmax计算出一个概率值！这个模型如下图所示：

<img src="http://blog.stupidme.me/wp-content/uploads/2018/10/softmax-nplm.png" style="width:500px">


正如前面所说，我们的word2vec并不需要一个完整的概率模型，我们只需要训练一个二分类器，从k个**噪声单词(noise words)**里面判别出正确的**目标词(target words)**。这`k`个噪声单词是随机选择出来的，这个技术叫做**负采样(negative sampling)**，因为选出来的一批词都是不是正确的target word。这个模型如下图所示：

<img src="http://blog.stupidme.me/wp-content/uploads/2018/10/nce-nplm.png" style="width:500px">

这样一来，我们要最大化的目标就是：

$$J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]$$

其中，$Q_\theta(D=1\vert w, h)$表示二分类逻辑回归在数据集D中的上下文h中包含目标$w_t$的概率。


### The classifier

上面说到了**负采样**。什么事负采样呢？其实就是**随机选取k个词语，和目标词组成负样本训练**。

现在我们回到斯坦福的教材上来。这里列出训练一个skip-gram模型的要点：

* 把目标词和上下文词组成的样本当做训练的**正样本(positive sample)**
* 随机选取一些词和目标词组成的样本当做训练的**负样本(negtive sample)**
* 使用logistic regression训练一个二分类器来区分两种情况
* regression的权重就是我们的**embedding**

word2vec需要的是训练一个binary logistic regression，给定一个目标$t$和候选上下文$c$的元组$(t,c)$，返回$c$正好是$t$的上下文词的概率：

$$P(+\vert t,c)$$

那么，$c$不是$t$的上下文词的概率就是：

$$P(-\vert t,c) = 1 - P(+\vert t,c)$$

那么分类器如何计算这个概率$P$呢？skip-gram模型有这样一个假设：**相近的词它们的嵌入表示也很近！**

也就是，我们可以把两个词语的嵌入表示的相似度，用来表示概率$P$！相似度就用我们上文说到的余弦相似度：

$$Similarity(t,c) \approx t\cdot c$$

当然，点积的结果并不是概率表示，我们需要用**logistic**或者叫**sigmoid**函数，把它转化为概率表示：

$$P(+\vert t,c) = \frac{1}{1+e^{-t\cdot c}}$$

那么：

$$P(-\vert t,c) = 1 - P(+\vert t,c) = \frac{e^{-t\cdot c}}{1+e^{-t\cdot c}}$$

上面的公式只是一个单词的概率，但是我们需要把整个window里面的单词计算进来。skip-gram模型还有一个假设：**所有的上下文单词之间是独立的**！

假设我们的`window_size = k`，于是有：

$$P(+\vert t,c_{1:k}) = \prod_{i=1}^k\frac{1}{1+e^{-t\cdot c_i}}$$

通常，我们会使用对数概率：

$$\log{P(+\vert t,c_{1:k})} = \sum_{i=1}^k\log{\frac{1}{1+e^{-t\cdot c_i}}}$$

### skip-gram模型的训练

为了训练这个word2vec，我们除了正样本，还需要负样本。实际上，负样本通常比正样本更多。一般用一个比率`k`来控制正负样本，如果`k=2`则说明，每一个正样本，对应2个负样本。这就是前面说的**负采样**技术。

构造负样本选择的词语(噪声词noise words)是根据一个频率来的：

$$p_\alpha(w) = \frac{count(w)^\alpha}{\sum_{w'}count(w')^\alpha}$$

其中，$\alpha$是一个比率，一般来说取值$\frac{3}{4} = 0.75$。

为什么需要这个比例呢？**这样可以让出现次数少的词被选择的可能性变大！**

举个例子，如果没有这个比率，假设$P(a) = 0.99$，$P(b) = 0.01$，加上这个比率之后：

$$P_\alpha(a) = 0.97$$

$$P_\alpha(b) = 0.03$$

可见，$b$得选择的概率从`0.01`提升到了`0.03`。

有了正负样本之后，我们的模型训练就有以下目标了：

* 最大化正样本的概率，也就是正样本的相似度最大化
* 最小化负样本的概率，也就是负样本的相似度最小化

在整个训练集上，用数学表示出上面的目标就是：

$$L(\theta) = \sum_{(t,c)\in +}\log P(+\vert t,c) + \sum_{(t,c)\in -}\log P(-\vert t,c)$$

如果从单个训练数据对来看(一个$(t,c)$ 对和$k$个噪声$n_1,n_2,\dots,n_k$)，就有：

$$L(\theta) = \log P(+\vert t,c) + \sum_{i=1}^k\log P(-\vert t,n_i)$$

概率P由simoid函数计算，有：

$$L(\theta) = \log\sigma(c\cdot t) + \sum_{i=1}^k\log\sigma(-n_i\cdot t)$$

展开，有：

$$L(\theta) = \log\frac{1}{1+e^{-c\cdot t}} + \sum_{i=1}^k\log\frac{1}{1+e^{c\cdot t}}$$

可以看出，最大化上面的目标，就是最大化正样本$c\cdot t$，同时最小化负样本$$n_i\cdot t$$。

有了上面的概率表示，那么我们就可以使用**交叉熵**作为损失函数，然后训练模型了！

值得注意的是，tensorflow里面把上面的两个过程合并了，合并在`tf.nn.nce_loss`这个函数里面。你可以看到tensorflow的教程里面的损失函数就是使用的`tf.nn.nce_loss`作为损失函数。但是你继续追踪源码就会发现，这个损失函数只不过是：

* 进行采样，计算出概率
* 使用**交叉熵**计算损失

可见，和我们上面的训练分析过程是吻合的！

### 两个权重矩阵W和C

还记得我们上面skip-gram模型训练的最后一个要点**regression的权重作为embedding**吗？

其实，word2vec训练之后会有两个权重矩阵，分别是**嵌入矩阵$W$**和**上下文矩阵C**，回顾一下这张图：

<img src="http://blog.stupidme.me/wp-content/uploads/2018/10/word2vec-skip-gram.png" style="width:500px">

上图中的$W$权重矩阵就是我们的**Embedding**矩阵，而$W'$权重矩阵就是我们的**Context**矩阵！

**如果我们要得到每一个单词的向量表示，只要从$W$中取出对应的行即可！**因为，训练的每一个单词，都是用one-hot编码的，直接和$W$相乘即可得到改词的向量表示。如果你对这一部分有疑问，请查看我之前的文章[自己动手实现word2vec(skip-gram模型)](https://juejin.im/post/5b986f296fb9a05d11176a15)。

所以，整个word2vec模型就是一个浅层的神经网络！

我们训练结束后，得到的两个矩阵$W$和$C$怎么用呢？一般情况下，我们不需要使用$C$，直接忽略掉即可。但是你也可以把两个矩阵**相加**，一起来表示新的`N`维嵌入表示，或者把他们**合并**，即$[W,C]$，用来创建一个新的`2*N`的嵌入表示。

当然，斯坦福的这个教程，后面还提到了词嵌入的可视化等信息，我就不写了。还是推荐大家去看原文，当然，我写的这个笔记中结合tensorflow那一部分也肯定可以解决你的一些疑惑。

## 推荐文章

1.[Vector Representations of Words](https://www.tensorflow.org/tutorials/representation/word2vec)  
2.[自己动手实现word2vec(skip-gram模型)](https://juejin.im/post/5b986f296fb9a05d11176a15)  
3.[Logistic Regression](https://luozhouyang.github.io/logistic_regression/)  
4.[Learning Word Embedding](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)  

