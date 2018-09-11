---
layout: post
title: "Attention机制"
summary: 你真的懂了Attention机制到底是什么吗？
featured-img: emile-perron-190221
---

前一阵子，看到一篇文章，讲**注意力机制(Attention mechanism)**的文章，非常不错。作者也是一个大佬。有兴趣的可以看原文：[Attetnion?Attetnion!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

最近，自己也想理一遍Attention机制，因此写了这篇笔记。

## seq2seq模型

**seq2seq** 模型生于语言模型领域，目标是将一个输入序列转化成另一个序列，他们可能有不一样的长度。

传统的seq2seq模型通常都是使用encoder-decoder架构：

* encoder处理输入序列并将这些信息压缩到一个 **固定大小** 的向量中，这个向量就是 **Context Vector**。当然有些地方会有其他称呼，但是本质是一样的。我们希望这个向量能够很好地表示整个输入序列的意义。
* decoder使用上面的context vector进行初始化，并且产生输出。早期的实现是，把encoder的最后一个输出作为decoder的初始状态。也就是说，我们的context vector是通过encoder的最后一个state生成的。

典型地，encoder和decoder都是一个RNN网络，比如使用 **LSTM** 或者 **GRU** 等标准RNN的变体。

这个经典的架构，如下图所示：
![seq2seq exmaple](/assets/art/encoder_decoder_example.png)

很显然，这种 **固定长度的** context vector有一个缺点，那就是不太适合处理长序列。通常的情况是，这个context vector会遗忘比较早的信息。而我们的 **注意力机制** 就是来解决这个问题的。

## attention机制

那么，attention机制有哪些不一样呢？

请看下图：
![attention](/assets/art/encoder_decoder_attention.png)

可以看到，这里的Context Vector和之前的不一样了。

之前的context vector由encoder的最后一个state产生，而这里的context vector与所有的encoder state都有联系。

**attention layer** 实际上是一个含有一个隐藏层的前馈网络。

我们的输入$$\mathbf{x}$$是一个长度为n的序列，输出$$\mathbf{y}$$是一个长度为m的序列：

$$\mathbf{x} = [x_1,x_2,\dots,x_n]$$

$$\mathbf{y} = [y_1,y_2,\dots,y_m]$$


encoder使用的是 **双向RNN**，对于每一个方向，都有一个hidden state，我们把两个hidden state合并在一起，作为encoder的state:

$$\boldsymbol{h}_i = [\overrightarrow{\boldsymbol{h}}_i^\top; \overleftarrow{\boldsymbol{h}}_i^\top]^\top, i=1,\dots,n$$

decoder的hidden state计算方法如下：

$$\boldsymbol{s}_t = f(\boldsymbol{s}_{t-1},y_{t-1},\mathbf{c}_t)$$

其中，$$\boldsymbol{s}_{t-1}$$ 是前一时刻的hidden state。$$y_{t-1}$$是前一时刻的输入.
$$\mathbf{c}_t$$是当前时刻的 **context vector**。

当前时刻的context vector计算方法如下：

$$\mathbf{c}_t=\sum_{i=1}^n\alpha_{t,i}\boldsymbol{h}_i$$

其中，$$\alpha_{t,i}$$代表$$y_t$$和$$x_i$$的对齐程度，实际上就是alignment score，就是一个权重，这个权重就是softmax得分。如下公式所示：

$$
\alpha_{t,i}  = \text{align}(y_t,x_i)  = \frac{\text{score}(\boldsymbol{s}_t,\boldsymbol{h}_{i})}{\sum_{i'=1}^n\text{score}(\boldsymbol{s}_t,\boldsymbol{h}_{i'})} 
$$

alignment model对于每一对输入$$x_i$$和$$y_t$$计算一个得分$$\alpha_{t,i}$$，那么这些$$\alpha$$的集合，就是定义了每一个输出被每一个hidden state影响了多少（也就是注意力在每一个hidden state上的分布）的权重矩阵。

那么，上面的分数函数score怎么计算呢？Bahdanau的论文表示，这个分数$$\alpha$$是被一个前馈网络参数化的，这个网络含有一个隐藏层。网络使用tanh激活函数：

$$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])$$

其中，$$\mathbf{v}_\alpha$$和$$\mathbf{W}_\alpha$$都是这个alignment model的权值矩阵。

传统的方法，处理输入序列和输出序列长度不同的情况，都要面临对齐这个问题。但是我们的attention机制，有一个很好的副产品，那就是 **自动进行了对齐**！

上述的attention叫做 **加性注意力机制(additive attention)**，[tensorflow/nmt](https://github.com/tensorflow/nmt)项目使用的就是这种attention机制。

Google 2017年的论文[Attention Is All You Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)提出的Transormer模型，使用的是另一种attention机制，叫做 **点积注意力机制(dot-product attention)**。

本文开始提到的文章，有一个attention机制的表格，非常全面：
![attention mechanism table](/assets/art/attention_mechanism_table.png)

所谓的 **additive attention** 意思是 **将输入序列的hidden state和输出序列的hidden state合并在一起**。即

$$\mathbf{W}_\alpha[\boldsymbol{s}_t;\boldsymbol{h}_i]$$

其中，$$\boldsymbol{s}_t$$就是输出序列的hidden state，$$\boldsymbol{h}_i$$就是输入序列的hidden state。

所谓 **dot-product attention** 意思是 **将输入序列的hidden state和输出序列的hidden state相乘**。见上表的公式。

所谓 **scaled dot-product attention** 是在 **dot-product attention** 的基础上，乘上一个 **缩放因子$$\frac{1}{\sqrt{n}}$$**，其中n代表模型的维度。优点在于这个缩放因子可以将函数值从softmax的饱和区拉回到非饱和区，这样可以防止梯度过小而很难学习。

**self-attention** 也叫做 **intra_attention**。
