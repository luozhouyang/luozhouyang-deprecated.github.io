---
layout: post
title: Logistic Regression笔记
summary: Logistics Regression你所需要知道的一切
featured-img: sleek
---

本文是阅读斯坦福经典教材 [Speech and Language Processing-Logistic Regression](https://web.stanford.edu/~jurafsky/slp3/5.pdf) 所做的笔记，推荐看原文。

Logistic Regression（以下简称逻辑回归“”）可以用于二分类问题和多分类问题(Multinomial logistic regression)。逻辑回归是一种分类算法，并不是回归算法。逻辑回归属于**判别式分类器（discriminative classifier）**，而朴素贝叶斯属于**生成式分类器（generative classifier）**。

## 判别式分类器和生成式分类器

为了区别这两种分类器，我们可以举一个简单的例子：区分照片里面的动物是猫还是狗。

Generative model的目标是，理解什么是猫什么事狗，然后做出判断。而Discriminative model则是紧紧学习怎样去区分这两种动物，而不是去学习它们是什么。

在数学上更直观的比较，首先看我们的niave Bayes分类公式：

$$\hat{c}=\mathop{\arg\max}_{c\in C}\overbrace{P(d\vert c)}^{\text{likelihood}} \overbrace{P(c)}^{\text{prior}}$$

对于generative model（例如naive Bayes）使用一个**似然(likelihood)**项来计算$P(c\vert d)$，这个项表示的是如何生成一个文档的特征，如果我们知道它是类别c的话。而对于discriminative model，它会尝试直接去计算$P(c\vert d)$。

## 基于概率的机器学习分类器的组成

基于概率的机器学习分类器有以下几个组成部分：

* 特征表示，即对每一个输入的表示
* 一个分类函数，用来估算当前输入的类别，例如**sigmoid**和**softmax**
* 一个目标函数，通常涉及在训练集上最小化误差，例如**交叉熵损失函数**
* 一个优化目标函数的算法，例如**SGD**

## Sigmoid

二分类逻辑回归的目标是训练一个分类器，可以做出二分类决策，sigmoid就是可行的方式之一。

逻辑回归通过从训练集学习两个参数$w$和$b$来做出决策。

逻辑回归的类别估算公式如下：

$$z=(\sum_{i=1}^nw_ix_i)+b$$

要学习的两个参数在上式也有直接体现。

在线性代数里面，通常把上面的加权和$\sum_{i=1}^nw_ix_i$用**点积(dot product)**来表示，所以上式等价于：

$$z=w\cdot x+b$$

那么得到的结果是一个浮点数，对于二分类为题，结果只有`0`和`1`两种，那我们怎么判断这个`z`是属于`0`类别还是`1`类别呢？

我们先看看sigmoid函数长什么样吧。

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

图像如下：

![sigmoid funtion](http://blog.stupidme.me/wp-content/uploads/2018/10/sigmoid.png)


可以看到，sigmoid函数的值域是(0,1)，并且是关于(0,0.5)对称的，所以很容易得到一个决策边界：

* `z<=0.5`时属于`0`类别
* `z>0.5`时属于`1`类别

sigmoid函数有很多很好的性质：

* 它的输入范围是$(-\inf,\inf)$，输出值范围是$(0,1)$，这就是天然的概率表示啊！
* 在`x=0`附近几乎是线性的，在非常负或者非常正的时候，变化不大

至此，我们可以计算类别`0`和类别`1`的概率：

$$P(y=1) = \sigma(w\cdot x +b) = \frac{1}{1+e^{-(w\cot x+b)}}$$

$$P(y=0) = 1-P(y=1) = 1-\sigma(w\cdot x+b) = 1-\frac{1}{1+e^{-(w\cdot x+b)}} = \frac{e^{-(w\cdot x+b)}}{1+e^{-(w\cdot x+b)}}$$

## cross-entropy损失函数

说到损失函数，你可能会想到**均方差损失(MSE)**：

$$L_{\text{MSE}}(\hat{y},y) = \frac{1}{1}(\hat{y}-y)^2$$

这个损失在线性回归里面用的很多，但是将它应用于概率分类的话，就变得难以优化了（主要是非凸性）。

**条件似然估计(conditional maximum likelihood estimation)**：选择参数$w$和$b$来最大化标签和训练数据之间（$P(y\vert x)$）的对数概率。

因为类别的分布是一个**伯努利分布(Bernoulli distribution)**，所以我们可以很容易写出：

$$p(y\vert x) = \hat{y}^y(1-\hat{y})^{1-y}$$

因为，当`y=1`时，$p(y\vert x)=\hat{y}$，当`y=0`时，$p(y\vert x)=1-\hat{y}$。

由此，可以得到**对数概率**：

$$\log p(y\vert x) = \log\left[\hat{y}^y(1-\hat{y})^{1-y}\right] = y\log\hat{y}+(1-y)\log(1-\hat{y})$$

我们的训练过程就是要最大化这个对数概率。如果对上式两边取负数，最大化问题就变成了最小化问题，即训练的目标就是最小化：

$$L_{CE}(\hat{y},y)=-\log p(y\vert x)=-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$$

又因为$\hat{y}=w\cdot x+b$，所以我们的**负对数似然损失**公式为：

$$L_{CE}=-[y\log\sigma(w\cdot x+b)+(1-y)\log(1-\sigma(w\cdot x+b))]$$

这也就是我们的**交叉熵损失(cross-entorpy loss)**，至于为什么是这个名称，因为上述公式就是：**$y$的概率分布和估计分布$\hat{y}$之间的交叉熵**。

所以，在整个数据集上，我们可以得到平均损失为：

$$Cost(w,b) = \frac{1}{m}\sum_{i=1}^mL_{CE}(\hat{y}^{(i)},y^{(i)}) = -\frac{1}{m}\sum_{i=1}^m y^{(i)}\log\sigma(w\cdot x^{(i)}) + (1-\hat{y}^{(i)})\log(1-\sigma(w\cdot x^{(i)}+b))$$

## 梯度下降

梯度下降的目标就是最小化损失，用公式表示就是：

$$\hat{\theta} = \mathop{\arg\min}_{\theta}\frac{1}{m}\sum_{i=1}^mL_{CE}(y^{(i)},x^{(i)};\theta)$$

对于我们的Logistic Regression，$\theta$就是$w$和$b$。

那么我们如何最小化这个损失呢？**梯度下降**就是一种寻找最小值的方式，它是通过倒数来得到函数的最快衰减方向来完成的。

对于逻辑回归的这个损失函数来说，它是**凸函数(convex function)**，所以它只有一个最小值，没有局部最小值，所以优化过程中肯定可以找到全局最小点。

举个二维的例子，感受一下这个过程，如下图所示：

![gredient decent example](http://blog.stupidme.me/wp-content/uploads/2018/10/gradient_decent_example.png)

可见，上述损失函数的优化过程就是**每次向着梯度的正方向移动一小步**!可以用公式表示如下：

$$w^{t+1} = w^t - \eta\frac{\partial}{\partial w}f(x;w)$$

上面 $\eta$ 决定了这个**一小步**是多少，也称作**学习率(learning rate)**。

上面的梯度 $\frac{\partial}{\partial w}f(x;w)$ 结果是一个常数。

如果是`N`维空间呢？那么梯度就是一个矢量了，如下所示：

![](http://blog.stupidme.me/wp-content/uploads/2018/10/nd_gradients.png)

那么，我们的参数更新就是：

$$\theta_{t+1}=\theta_t - \eta\nabla L(f(x;\theta),y)$$

