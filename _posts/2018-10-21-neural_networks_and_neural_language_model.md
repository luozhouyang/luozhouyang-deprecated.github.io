---
layout: post
title: "神经网络和神经语言模型"
summary: 关于神经语言模型你所需要知道的一切
featured-img: emile-perron-190221
---

斯坦福教程[Sppech and Language Processing - Neural Networks and Nerual Language Models](https://web.stanford.edu/~jurafsky/slp3/7.pdf)笔记。

# 神经网络

神经网络的概念很早就有了。现代神经网络是由一些小的计算单元组成的网络，其中每一层接收一个向量（更学术一点应该叫做张量）作为输入，并且产生一个向量输出。本文介绍将神经网络用于分类。这只是一个**前馈网络(feed-forward network)**，因为这个网络逐层处理输入并且产生输出。这种类似的技术现在一般叫做**深度学习(deep learning)**，因为这些网络通常是**深的(有多个层)**。

**神经网络和**和**Logistic Regression**有很多相同的数学公式，但是神经网络要更加强大，事实上，一个最小化的神经网络（只包含一个隐藏层），可以学习任意函数！

## 神经单元

我们把神经网络的最小单元叫做神经单元（Unit）。每个单元都是对输入进行一些计算，然后产生一个输出。核心思想是，对输入进行一个加权和(weighted sum)，然后加上一个偏置(bias)：

$$z = b + \sum_i w_ix_i$$

在线性代数里，加权和表示成点积：

$$z = w\cdot x + b $$

最后，我们把上面的结果通过一个**非线性函数**或者叫做**激活函数**。

我们熟悉的**sigmoid**函数或者叫做**logistic**函数就是一个激活函数：

$$\text{sigmoid}(z) = \sigma(z) = \frac{1}{1+e^{-z}}$$

图像如下：

![sigmoid](http://blog.stupidme.me/wp-content/uploads/2018/10/sigmoid.png)

举个例子，假设激活函数是**sigmoid**，则神经单元的输出为：

$$ y = \sigma(z) = \sigma(w\cdot x + b ) = \frac{1}{1+\exp{-(w\cot x +b)}}$$

所以，整个神经单元的数据处理过程可以表示为下图：

![neural unit](http://blog.stupidme.me/wp-content/uploads/2018/10/neural_unit.png)

但是，实际上**sigmoid**激活函数已经不再推荐使用了，你应该总是使用**tanh**激活函数来替代**sigmoid**。**tanh**激活函数是**sigmoid**的一个变种，它的值域是$(-1,1)$并且关于$(0,0)$对称，它的公式如下：

$$\text{tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

还有一个最常使用的激活函数**ReLU(rectified linear function)**，它的公式如下：

$$\text{relu}(z) = \max(x,0)$$

下面是分别是**tanh**和**ReLU**激活函数的图像：

<!-- <img src="http://blog.stupidme.me/wp-content/uploads/2018/10/tanh_and_relu.png" style="width:600px"> -->
![](http://blog.stupidme.me/wp-content/uploads/2018/10/tanh_and_relu.png)

## 异或(XOR)问题

单个神经单元可能无法解决一些很简单的函数，比如`AND`、`OR`和`XOR`。

**感知机(perceptron)**是一个简单的神经单元，没有激活函数。它可以**逻辑与(AND)**和**逻辑或(OR)**运算，但它无法进行**逻辑异或(XOR)**运算。

感知机是一个线性分类器，它的决策边界是一条直线：

$$w_1x_1+w_2x_2+b = 0$$

它能够进行`AND`和`OR`运算，而不能进行`XOR`运算，从下图可以看出来，`XOR`不是线性可分的：

![XOR](http://blog.stupidme.me/wp-content/uploads/2018/10/xor.png)

那么该怎么办呢？答案是**使用神经网络**！

Goodfellow等人在2016年使用一个两层的神经网络计算出`XOR`问题，使用ReLU激活函数。为什么神经网络可以呢？**神经网络的隐藏层把输入重新表示了**！如下图所示：

![xor_nn_solution](http://blog.stupidme.me/wp-content/uploads/2018/10/xor_nn_solution.png)