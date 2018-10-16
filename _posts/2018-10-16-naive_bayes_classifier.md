---
layout: post
title: Naive Bayes分类器
summary: Naive Bayes分类器你所需要知道的一切
featured-img: ruben-santander-69158
---

## Naive Bayes分类器

Naive Bayes是一个概率分类器，也就是说，在文档d中，返回所有类别c中后验概率最大的类别$\hat{c}$:

$$\hat{c}=\text{argmax}P(c\vert d)$$

回顾一下贝叶斯法则：

$$P(x\vert y)=\frac{P(y\vert x)P(x)}{P(y)}$$

它把任何**条件概率**转化成了三个概率。

其中，$P(y)$是**先验概率**或者**边缘概率**。

贝叶斯法则可以从条件概率的定义推导，过程如下：

$$P(A\vert B) = \frac{P(A\cap B)}{P(B)}$$

又，

$$P(A\vert B)P(B) = P(A\cap B) = P(B\vert A)P(A)$$

所以，

$$P(A\vert B) = \frac{P(B\vert A)P(A)}{P(B)}$$

上面第二个公式又叫做**概率乘法法则**。

回到之前的$\hat{c}$，那么此时有：

$$\hat{c}=\text{argmax}P(c\vert d)=\text{argmax}\frac{P(d\vert c)P(c)}{P(d)}$$

因为$P(d)$对于任何$c$都是一个不变的值，所以可以省去：

$$\hat{c}=\text{argmax}P(c\vert d)=\text{argmax}P(d\vert c)P(c)$$

上式，$P(d\vert c)$叫做**似然(likelihood)**，$P(c)$即**先验概率(prior probability)**。

此时，假设文档$d$由`n`个特征组成，则有：

$$\hat{c}=\text{argmax}\overbrace{P(f_1,f_2,\dots,f_n\vert c)}^{\text{likelihood}}\ \overbrace{P(c)}^{\text{prior}}$$

要计算上面的**似然**，需要很多的参数和很大的训练集，这个很难实现。

朴素贝叶斯有两个假设：

* 位置无关
* $P(f_i\vert c)$条件独立，也称**朴素贝叶斯假设**

所以上式可以简化为：

$$P(f_1,f_2,\dots,f_n\vert c)=P(f_1\vert c)P(f_2\vert c)\dots P(f_n\vert c)$$

即：

$$C_{NB}=\text{argmax}P(c)\prod_{f\in F}P(f\vert c)$$

**词袋模型(bag of words)**不考虑词语的位置，把词语出现的频次当做特征，于是有：

$$C_{NB}=\text{argmax}P(c)\prod_{i\in positions}P(w_i\vert c)$$

为了避免数值下溢和提高计算速度，通常使用对数形式：

$$c_{NB}=\text{argmax}\log{P(c)+\sum_{i\in positions}\log{P(w_i\vert c)}}$$

## 训练朴素贝叶斯分类器

为了知道$P(c)$和$P(f_i\vert c)$，我们还是使用**最大似然估计(MLE)**。

有：
$$\hat{P}(c)=\frac{N_c}{N_{doc}}$$

$$\hat{P}(w_i\vert c)=\frac{count(w_i,c)}{\sum_{w\in V}count(w,c)}$$

为了避免某个概率值为0，我们使用**拉普拉斯平滑(Laplace smooth or add-one smooth)**：

$$\hat{P}(w_i\vert c)=\frac{count(w_i,c)+1}{\sum_{w\in V}(count(w,c)+1)}=\frac{count(w_i,c)+1}{(\sum_{w\in V}count(w,c))+\vert V\vert}$$

对于**unknown word**怎么处理呢？答案是：**直接从测试数据集中移除这些词，不计算概率**！

## 评估
TODO
### Precision

### Recall

### F-measure
