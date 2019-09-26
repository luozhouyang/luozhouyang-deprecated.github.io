---
layout: post
title: "Attention机制"
summary: 你真的懂了Attention机制到底是什么吗？
featured-img: emile-perron-190221
---

前一阵子，看到一篇文章，讲**注意力机制(Attention mechanism)**的文章，非常不错。作者也是一个大佬。有兴趣的可以看原文：[Attetnion?Attetnion!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

最近，自己也想理一遍Attention机制，因此写了这篇笔记。

其实谷歌爸爸的教程提供了非常清晰的介绍：

* [BahdanauAttention - NMT with attention](https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention)
* [MultiHeadAttention - Transformer](https://www.tensorflow.org/beta/tutorials/text/transformer)

仔细看完两个教程，我们可以归纳一下要点：

* Attention机制可以抽象为`Q`、`K`、`V`三个张量的计算
* 不同Attention机制的区别在于`Q`、`K`、`V`三个张量的选择，和其中`score`函数的选择

我们分别从`Q`、`K`、`V`张量的选择和`score`函数的选择两个角度来分别分析Attention机制的异同。


## `Q`、`K`、`V`张量的选择

我准备用`Transformer`和`NMT with Attention`两个经典的网络来分析，就如谷歌爸爸的教程一样。

### Transformer的Attention机制

我们首先看一下**Transformer**的`Scaled dot-product Attention`的实现：

```python
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
```

代码十分清晰，我们可以看到，`Scaled dot-product attention`的三个参数`Q`、`K`、`V`刚好一一对应我们的归纳的第一点的三个张量，并且`score`函数，选择的是`dot-product`，然后乘以一个**缩放因子**`1/tf.math.sqrt(dk)`。所以有了`Scaled dot-product Attention`的名字。

那么，Transformer里面还有`MultiHeadAttention`和`Self-Attention`的概念，它们和`Scaled dot-product Attention`有什么联系呢？

其实很简单 ：

* `Self Attention`就是一种特殊的`Scaled dot-product Attention`，它的`Q == K == V`！
* `MultiHeadAttention`是分成多头之后，每个头进行`Scaled dot-product Attention`！

注：
> 在Transformer里面，`Scaled dot-product Attention`不仅仅表现为`Self Attention`，对于`Encoder`和`Decoder`之间的Attention计算，也使用的是`Scaled dot-product  Attention`，只不过它的`Q`、`K`、`V`并不是完全一样的，我个人在Transformer的语境下更倾向于称之为`Context Attention`。

Transformer的Attention机制你应该非常清楚了！


### NMT的Attention机制

你可能还会想到另外一个经典的NMT使用的Attention机制。实际上，在NMT里有两种比较常用的Attention，在tensorflow里称为：

* `BahdanauAttention`
* `LuongAttention`

它们都是用人名命名的Attention机制。对于`tensorflow 1.x`的源代码，你可以看到具体的实现。

我们在这里只分析前一种`BahdanauAttention`，搞懂了这个，你可以对照源码去分析一下`LuongAttention`，就当作练习！

开头提到的**nmt with attention**一文中使用的是`BahdanauAttention`，这种Attention有什么特点呢？

它的代码实现如下：

```python
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```

我们发现，这种attention只接受两个参数，分别是`Q`、`V`两个张量。我们刚刚说Attention是`Q`、`K`、`V`三个张量的计算，那么它的`K`去哪里了呢？

我们首先理解一下一般的`Q`、`K`、`V`之间的计算逻辑：

* 用`Q`和`K`进行计算，得到`score`，这一步也就是我们所说的`score`函数
* 对`score`函数进行`softmax`归一化，得到`attention weights`
* 用`attention_weights`和`V`进行计算，得到输出

这三个步骤是黄金法则，Transformer里面的`Scaled dot-product Attention`是这么计算的，我们这里的`BahdanauAttention`也是。

在`BahdanauAttention`里：

* `score`函数的`values`就是步骤一的`K`
* `score`函数的`hidden_with_time_axis`就是步骤一中的`Q`
* `score`函数就是`K`和`Q`线性变换后，相加，经过`tanh`激活，再次线性变换得到的，对应步骤二
* `attention_weights`就是对`score`进行`softmax`归一化，也是步骤二
* 输出就是`attention_weights`和`V`相乘，对应步骤三

看吧，其实不同的Attention机制，都是按照这个规律来的。

## `score`函数的选择

上面一小节我们看到了`Q`、`K`、`V`的选择对应的不同的Attention机制，这里我们再介绍一点关于`score`函数的选择。

其实上面一小节，我也介绍了两种Attention机制的`score`函数的不同，这里我就用文章开头的博客链接的一张图，作为展示：

![attention mechanism table](/assets/art/attention_mechanism_table.png)


## 总结

这些都是我个人总结出来的规律。从这个角度来理解Attention，发现特别轻松。

希望对大家有帮助。
