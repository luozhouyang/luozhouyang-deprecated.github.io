---
layout: page
title: 个人简历
permalink: /resume/
---

## 基本信息
罗周杨， 男，1995年生。

学校： [上海大学](http://www.shu.edu.cn/)，本科。

专业； `通信学院 | 电子信息工程`

手机： **13381536323**

邮箱： **zhouyang.luo@gmail.com**

## 工作经历

* `2018年12月——至今` `算法工程师`@[ifchange](https://www.ifchange.com)
  
  使用深度学习（NLP）模型，改善搜索和推荐效果。主要设涉及NLP中的文本匹配任务。

  主要内容：
  * 研究DSSM网络，用于计算query和doc的相似度
  * 研究MatchPyramid网络，用于计算query和doc的相似度
  * 基于Transformer和BERT构建模型，用于计算query和doc的相似度


* `2017年7月——2018年12月` `软件工程师`@[51job](https://www.51job.com)
  
  负责地址短文本的纠错任务。使用经典的seq2seq模型。

  主要内容：
  > 受[ tensorflow/nmt ](https://github.com/tensorflow/nmt)项目启发，通过经典的`seq2seq`架构，使用多层双向`RNN（LSTM/GRU）`实现一个经典的 `encoder-decoder` 模型，用于文本纠错（已经有大量人工标注的训练数据）。测试集的**正确率(Accuracy)**达到`71%`左右（其他指标当时未计算）。训练模型之后，导出为tensorflow的`SavedModel`格式，使用 `tensorflow serving`部署模型，平均响应时间为`50ms`左右。为了提高`分词`效果，实现了一个`Bi-LSTM + CRF`的模型，用来分词。

## 个人技能

* 熟悉Java语言，4年使用经验。
* 熟悉Python，目前为主力语言。
* 熟悉NLP领域的多种深度学习算法，熟悉**神经网络**用于文本序列的建模，熟悉**Attention**机制，熟悉**Transformer**和**BERT**模型。
  

## 开源

* [luozhouyang/transformer](https://github.com/luozhouyang/transformer)

  Transformer的Tensorflow实现。

* [luozhouyang/dssm](https://github.com/luozhouyang/dssm)

  Deep Semantic Similarity Model的Tensorflow实现。

* [luozhouyang/matchpyramid](https://github.com/luozhouyang/matchpyramid)

  MatchPyramid的Tensorflow实现。

* [naivenmt/datasets](https://github.com/naivenmt/datasets)

  Tensorflow Data API实现的常用的高效的数据输入管道工具包。


## 博客

* [Transformer的Pytorch实现](https://luozhouyang.github.io/transformer/)
* [更多](https://luozhouyang.github.io)

