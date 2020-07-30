---
layout: post
title: tf.keras加载预训练模型：以BERT为例
summary: 加载预训练模型其实很简单
featured-img: shane-rounce-205187
---

`BERT`的出现拉开了预训练语言模型的序幕。

假设有这样一个场景：

> 你自己实现了一个BERT模型，你的实现和Google的实现相差比较大，比如Layer的层次组织不同，比如Google用的是`tf.estimator`实现的，而你使用`tf.keras`实现。但是你想直接加载Google提供的预训练的BERT模型。

你有什么思路呢？

显然，你会尝试阅读`tensorflow`的官方文档，或者在GitHub上找实现了加载预训练模型的仓库，然后看代码是如何实现加载过程的。

然后很快你会发现，`tensorflow`官方好像没有很好的文档或者具体的例子，GitHub上的实现代码不是很直接，各种函数跳来跳去。

然后你就会想，加载一个预训练模型，难道不是应该很简单很清晰的吗？

事实上，确实如此，加载一个预训练模型确实很简单，也应该做到很清晰。


## 模型是什么

模型不是什么和高大上的东西，它很简单：**就是各种变量的集合**，然后还会附带一些元信息。下载的预训练模型还会包含数据文件，例如配置文件、词典等等。不过最重要的还是这些变量。

你自己实现的BERT模型，构建出网络之后，它会有自己的变量。预训练的模型，也保存了很多的变量。

如果你实现的BERT模型和Google实现的模型是一样的，那么这两者在变量上一定是一样的，或者说是有一一对应关系的。

所以，所谓的加载预训练模型，就是把**Google预训练的模型中的变量，设置到你自己实现的模型对应的变量上去！**

## 预训练模型是什么样的

首先，假设我们以及下载好了一个`BERT`模型: `/tmp/chinese_L-12_H-768_A-12`。它包含以下内容：

```bash
bert_config.json                    bert_model.ckpt.index               vocab.txt
bert_model.ckpt.data-00000-of-00001 bert_model.ckpt.meta
```
其中：

* `bert_config.json`是模型的配置文件，决定了网络
* `vocab.txt`是训练该模型使用的词典文件
* `bert_model.ckpt.*`就是训练的模型变量权重

那么我们如何知道它包含哪些变量呢？变量名字和值又是多少呢？

我们可以很容易的打印出这些变量:

```python
import tensorflow as tf

for v in tf.train.list_variables('/tmp/chinese_L-12_H-768_A-12/bert_model.ckpt'):
    print(v)
```

我们可以得到以下输出:

```bash
('bert/embeddings/LayerNorm/beta', [768])
('bert/embeddings/LayerNorm/gamma', [768])
('bert/embeddings/position_embeddings', [512, 768])
('bert/embeddings/token_type_embeddings', [2, 768])
('bert/embeddings/word_embeddings', [21128, 768])
('bert/encoder/layer_0/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_0/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_0/attention/output/dense/bias', [768])
('bert/encoder/layer_0/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_0/attention/self/key/bias', [768])
('bert/encoder/layer_0/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_0/attention/self/query/bias', [768])
('bert/encoder/layer_0/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_0/attention/self/value/bias', [768])
('bert/encoder/layer_0/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_0/intermediate/dense/bias', [3072])
('bert/encoder/layer_0/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_0/output/LayerNorm/beta', [768])
('bert/encoder/layer_0/output/LayerNorm/gamma', [768])
('bert/encoder/layer_0/output/dense/bias', [768])
('bert/encoder/layer_0/output/dense/kernel', [3072, 768])
('bert/encoder/layer_1/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_1/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_1/attention/output/dense/bias', [768])
('bert/encoder/layer_1/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_1/attention/self/key/bias', [768])
('bert/encoder/layer_1/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_1/attention/self/query/bias', [768])
('bert/encoder/layer_1/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_1/attention/self/value/bias', [768])
('bert/encoder/layer_1/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_1/intermediate/dense/bias', [3072])
('bert/encoder/layer_1/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_1/output/LayerNorm/beta', [768])
('bert/encoder/layer_1/output/LayerNorm/gamma', [768])
('bert/encoder/layer_1/output/dense/bias', [768])
('bert/encoder/layer_1/output/dense/kernel', [3072, 768])
('bert/encoder/layer_10/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_10/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_10/attention/output/dense/bias', [768])
('bert/encoder/layer_10/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_10/attention/self/key/bias', [768])
('bert/encoder/layer_10/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_10/attention/self/query/bias', [768])
('bert/encoder/layer_10/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_10/attention/self/value/bias', [768])
('bert/encoder/layer_10/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_10/intermediate/dense/bias', [3072])
('bert/encoder/layer_10/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_10/output/LayerNorm/beta', [768])
('bert/encoder/layer_10/output/LayerNorm/gamma', [768])
('bert/encoder/layer_10/output/dense/bias', [768])
('bert/encoder/layer_10/output/dense/kernel', [3072, 768])
('bert/encoder/layer_11/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_11/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_11/attention/output/dense/bias', [768])
('bert/encoder/layer_11/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_11/attention/self/key/bias', [768])
('bert/encoder/layer_11/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_11/attention/self/query/bias', [768])
('bert/encoder/layer_11/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_11/attention/self/value/bias', [768])
('bert/encoder/layer_11/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_11/intermediate/dense/bias', [3072])
('bert/encoder/layer_11/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_11/output/LayerNorm/beta', [768])
('bert/encoder/layer_11/output/LayerNorm/gamma', [768])
('bert/encoder/layer_11/output/dense/bias', [768])
('bert/encoder/layer_11/output/dense/kernel', [3072, 768])
('bert/encoder/layer_2/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_2/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_2/attention/output/dense/bias', [768])
('bert/encoder/layer_2/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_2/attention/self/key/bias', [768])
('bert/encoder/layer_2/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_2/attention/self/query/bias', [768])
('bert/encoder/layer_2/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_2/attention/self/value/bias', [768])
('bert/encoder/layer_2/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_2/intermediate/dense/bias', [3072])
('bert/encoder/layer_2/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_2/output/LayerNorm/beta', [768])
('bert/encoder/layer_2/output/LayerNorm/gamma', [768])
('bert/encoder/layer_2/output/dense/bias', [768])
('bert/encoder/layer_2/output/dense/kernel', [3072, 768])
('bert/encoder/layer_3/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_3/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_3/attention/output/dense/bias', [768])
('bert/encoder/layer_3/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_3/attention/self/key/bias', [768])
('bert/encoder/layer_3/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_3/attention/self/query/bias', [768])
('bert/encoder/layer_3/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_3/attention/self/value/bias', [768])
('bert/encoder/layer_3/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_3/intermediate/dense/bias', [3072])
('bert/encoder/layer_3/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_3/output/LayerNorm/beta', [768])
('bert/encoder/layer_3/output/LayerNorm/gamma', [768])
('bert/encoder/layer_3/output/dense/bias', [768])
('bert/encoder/layer_3/output/dense/kernel', [3072, 768])
('bert/encoder/layer_4/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_4/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_4/attention/output/dense/bias', [768])
('bert/encoder/layer_4/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_4/attention/self/key/bias', [768])
('bert/encoder/layer_4/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_4/attention/self/query/bias', [768])
('bert/encoder/layer_4/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_4/attention/self/value/bias', [768])
('bert/encoder/layer_4/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_4/intermediate/dense/bias', [3072])
('bert/encoder/layer_4/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_4/output/LayerNorm/beta', [768])
('bert/encoder/layer_4/output/LayerNorm/gamma', [768])
('bert/encoder/layer_4/output/dense/bias', [768])
('bert/encoder/layer_4/output/dense/kernel', [3072, 768])
('bert/encoder/layer_5/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_5/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_5/attention/output/dense/bias', [768])
('bert/encoder/layer_5/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_5/attention/self/key/bias', [768])
('bert/encoder/layer_5/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_5/attention/self/query/bias', [768])
('bert/encoder/layer_5/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_5/attention/self/value/bias', [768])
('bert/encoder/layer_5/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_5/intermediate/dense/bias', [3072])
('bert/encoder/layer_5/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_5/output/LayerNorm/beta', [768])
('bert/encoder/layer_5/output/LayerNorm/gamma', [768])
('bert/encoder/layer_5/output/dense/bias', [768])
('bert/encoder/layer_5/output/dense/kernel', [3072, 768])
('bert/encoder/layer_6/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_6/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_6/attention/output/dense/bias', [768])
('bert/encoder/layer_6/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_6/attention/self/key/bias', [768])
('bert/encoder/layer_6/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_6/attention/self/query/bias', [768])
('bert/encoder/layer_6/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_6/attention/self/value/bias', [768])
('bert/encoder/layer_6/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_6/intermediate/dense/bias', [3072])
('bert/encoder/layer_6/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_6/output/LayerNorm/beta', [768])
('bert/encoder/layer_6/output/LayerNorm/gamma', [768])
('bert/encoder/layer_6/output/dense/bias', [768])
('bert/encoder/layer_6/output/dense/kernel', [3072, 768])
('bert/encoder/layer_7/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_7/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_7/attention/output/dense/bias', [768])
('bert/encoder/layer_7/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_7/attention/self/key/bias', [768])
('bert/encoder/layer_7/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_7/attention/self/query/bias', [768])
('bert/encoder/layer_7/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_7/attention/self/value/bias', [768])
('bert/encoder/layer_7/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_7/intermediate/dense/bias', [3072])
('bert/encoder/layer_7/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_7/output/LayerNorm/beta', [768])
('bert/encoder/layer_7/output/LayerNorm/gamma', [768])
('bert/encoder/layer_7/output/dense/bias', [768])
('bert/encoder/layer_7/output/dense/kernel', [3072, 768])
('bert/encoder/layer_8/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_8/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_8/attention/output/dense/bias', [768])
('bert/encoder/layer_8/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_8/attention/self/key/bias', [768])
('bert/encoder/layer_8/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_8/attention/self/query/bias', [768])
('bert/encoder/layer_8/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_8/attention/self/value/bias', [768])
('bert/encoder/layer_8/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_8/intermediate/dense/bias', [3072])
('bert/encoder/layer_8/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_8/output/LayerNorm/beta', [768])
('bert/encoder/layer_8/output/LayerNorm/gamma', [768])
('bert/encoder/layer_8/output/dense/bias', [768])
('bert/encoder/layer_8/output/dense/kernel', [3072, 768])
('bert/encoder/layer_9/attention/output/LayerNorm/beta', [768])
('bert/encoder/layer_9/attention/output/LayerNorm/gamma', [768])
('bert/encoder/layer_9/attention/output/dense/bias', [768])
('bert/encoder/layer_9/attention/output/dense/kernel', [768, 768])
('bert/encoder/layer_9/attention/self/key/bias', [768])
('bert/encoder/layer_9/attention/self/key/kernel', [768, 768])
('bert/encoder/layer_9/attention/self/query/bias', [768])
('bert/encoder/layer_9/attention/self/query/kernel', [768, 768])
('bert/encoder/layer_9/attention/self/value/bias', [768])
('bert/encoder/layer_9/attention/self/value/kernel', [768, 768])
('bert/encoder/layer_9/intermediate/dense/bias', [3072])
('bert/encoder/layer_9/intermediate/dense/kernel', [768, 3072])
('bert/encoder/layer_9/output/LayerNorm/beta', [768])
('bert/encoder/layer_9/output/LayerNorm/gamma', [768])
('bert/encoder/layer_9/output/dense/bias', [768])
('bert/encoder/layer_9/output/dense/kernel', [3072, 768])
('bert/pooler/dense/bias', [768])
('bert/pooler/dense/kernel', [768, 768])
('cls/predictions/output_bias', [21128])
('cls/predictions/transform/LayerNorm/beta', [768])
('cls/predictions/transform/LayerNorm/gamma', [768])
('cls/predictions/transform/dense/bias', [768])
('cls/predictions/transform/dense/kernel', [768, 768])
('cls/seq_relationship/output_bias', [2])
('cls/seq_relationship/output_weights', [2, 768])
```

如果你熟悉`BERT`模型，那么一眼就可以看懂这些变量是指的什么。

可以看到，这个模型有`12`层`Encoder`，每个`Encoder`层里面有`MultiHeadAttention`层，包含了`MLM`和`NSP`两个头，`Embedding`层包含`token embedding`、`position embedding`和`token type embedding`。



## 你的BERT模型是什么样的

接下来，我们实现的`BERT`模型是什么样子的呢？

我个人实现的`BERT`在 [luozhouyang/transformers-keras](https://github.com/luozhouyang/transformers-keras)。

要构建出BERT模型，我们需要给模型设置参数。我的代码实现，需要这些参数：

```python
config = {
    'num_layers': self.num_layers,
    'hidden_size': self.hidden_size,
    'num_attention_heads': self.num_attention_heads,
    'intermediate_size': self.intermediate_size,
    'activation': tf.keras.activations.serialize(self.activation),
    'vocab_size': self.vocab_size,
    'max_positions': self.max_positions,
    'type_vocab_size': self.type_vocab_size,
    'dropout_rate': self.dropout_rate,
    'epsilon': self.epsilon,
    'stddev': self.stddev,
}
```

我们再看一眼预训练模型的参数有哪些：

```python
{
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 21128
}
```

虽然参数不一样，但是可以很容易一一对应上去。所以，在用我们自己的代码创建网络之前，我们需要按照预训练模型的配置，构造出适合我们模型的参数：

```python
def _map_model_config(self, pretrain_config_file):
    with open(pretrain_config_file, mode='rt', encoding='utf8') as fin:
        config = json.load(fin)

    model_config = {
        'vocab_size': config['vocab_size'],
        'activation': config['hidden_act'],
        'max_positions': config['max_position_embeddings'],
        'hidden_size': config['hidden_size'],
        'type_vocab_size': config['type_vocab_size'],
        'intermediate_size': config['intermediate_size'],
        'dropout_rate': config['hidden_dropout_prob'],
        'stddev': config['initializer_range'],
        'num_layers': config['num_hidden_layers'],
        'num_attention_heads': config['num_attention_heads'],
    }

    return model_config
```

有了正确的参数，那么我就可以用我们自己的代码构造出`BERT`模型：

```python
def build_pretraining_bert_model(model_config):
    max_sequence_length = model_config.get('max_positions', 512)
    input_ids = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='segment_ids')

    inputs = (input_ids, segment_ids, input_mask)
    bert = Bert4PreTraining(**model_config)
    outputs = bert(inputs)

    predictions = tf.keras.layers.Lambda(lambda x: x[0], name='predictions')(outputs[0])
    relations = tf.keras.layers.Lambda(lambda x: x[1], name='relations')(outputs[1])

    model = tf.keras.Model(inputs=inputs, outputs=[predictions, relations])
    lr = model_config.get('learning_rate', 3e-5)
    epsilon = model_config.get('epsilon', 1e-12)
    clipnorm = model_config.get('clipnorm', 1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=clipnorm),
        loss={
            'predictions': MaskedSparseCategoricalCrossentropy(mask_id=0, from_logits=True, name='pred_loss'),
            'relations': tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='rel_loss'),
        },
        metrics={
            'predictions': [
                MaskedSparseCategoricalAccuracy(mask_id=0, from_logits=True, name='pred_acc'),
            ],
            'relations': [
                tf.keras.metrics.CategoricalAccuracy(name='rel_acc'),
            ]
        })
    model.summary()
    return model
```

然后，看看我们的模型包含哪些变量：

```python
for v in model.trainable_variables:
    print(v.name)
```

可以看到：

```bash
bert/main/embedding/weight:0
bert/main/embedding/position_embedding/embeddings:0
bert/main/embedding/token_type_embedding/embeddings:0
bert/main/embedding/layer_normalization/gamma:0
bert/main/embedding/layer_normalization/beta:0
bert/main/encoder/layer_0/mha/query/kernel:0
bert/main/encoder/layer_0/mha/query/bias:0
bert/main/encoder/layer_0/mha/key/kernel:0
bert/main/encoder/layer_0/mha/key/bias:0
bert/main/encoder/layer_0/mha/value/kernel:0
bert/main/encoder/layer_0/mha/value/bias:0
bert/main/encoder/layer_0/mha/dense/kernel:0
bert/main/encoder/layer_0/mha/dense/bias:0
bert/main/encoder/layer_0/attn_layer_norm/gamma:0
bert/main/encoder/layer_0/attn_layer_norm/beta:0
bert/main/encoder/layer_0/intermediate/dense/kernel:0
bert/main/encoder/layer_0/intermediate/dense/bias:0
bert/main/encoder/layer_0/dense/kernel:0
bert/main/encoder/layer_0/dense/bias:0
bert/main/encoder/layer_0/inter_layer_norm/gamma:0
bert/main/encoder/layer_0/inter_layer_norm/beta:0
bert/main/encoder/layer_1/mha/query/kernel:0
bert/main/encoder/layer_1/mha/query/bias:0
bert/main/encoder/layer_1/mha/key/kernel:0
bert/main/encoder/layer_1/mha/key/bias:0
bert/main/encoder/layer_1/mha/value/kernel:0
bert/main/encoder/layer_1/mha/value/bias:0
bert/main/encoder/layer_1/mha/dense_1/kernel:0
bert/main/encoder/layer_1/mha/dense_1/bias:0
bert/main/encoder/layer_1/attn_layer_norm/gamma:0
bert/main/encoder/layer_1/attn_layer_norm/beta:0
bert/main/encoder/layer_1/intermediate/dense/kernel:0
bert/main/encoder/layer_1/intermediate/dense/bias:0
bert/main/encoder/layer_1/dense/kernel:0
bert/main/encoder/layer_1/dense/bias:0
bert/main/encoder/layer_1/inter_layer_norm/gamma:0
bert/main/encoder/layer_1/inter_layer_norm/beta:0
bert/main/encoder/layer_2/mha/query/kernel:0
bert/main/encoder/layer_2/mha/query/bias:0
bert/main/encoder/layer_2/mha/key/kernel:0
bert/main/encoder/layer_2/mha/key/bias:0
bert/main/encoder/layer_2/mha/value/kernel:0
bert/main/encoder/layer_2/mha/value/bias:0
bert/main/encoder/layer_2/mha/dense_2/kernel:0
bert/main/encoder/layer_2/mha/dense_2/bias:0
bert/main/encoder/layer_2/attn_layer_norm/gamma:0
bert/main/encoder/layer_2/attn_layer_norm/beta:0
bert/main/encoder/layer_2/intermediate/dense/kernel:0
bert/main/encoder/layer_2/intermediate/dense/bias:0
bert/main/encoder/layer_2/dense/kernel:0
bert/main/encoder/layer_2/dense/bias:0
bert/main/encoder/layer_2/inter_layer_norm/gamma:0
bert/main/encoder/layer_2/inter_layer_norm/beta:0
bert/main/encoder/layer_3/mha/query/kernel:0
bert/main/encoder/layer_3/mha/query/bias:0
bert/main/encoder/layer_3/mha/key/kernel:0
bert/main/encoder/layer_3/mha/key/bias:0
bert/main/encoder/layer_3/mha/value/kernel:0
bert/main/encoder/layer_3/mha/value/bias:0
bert/main/encoder/layer_3/mha/dense_3/kernel:0
bert/main/encoder/layer_3/mha/dense_3/bias:0
bert/main/encoder/layer_3/attn_layer_norm/gamma:0
bert/main/encoder/layer_3/attn_layer_norm/beta:0
bert/main/encoder/layer_3/intermediate/dense/kernel:0
bert/main/encoder/layer_3/intermediate/dense/bias:0
bert/main/encoder/layer_3/dense/kernel:0
bert/main/encoder/layer_3/dense/bias:0
bert/main/encoder/layer_3/inter_layer_norm/gamma:0
bert/main/encoder/layer_3/inter_layer_norm/beta:0
bert/main/encoder/layer_4/mha/query/kernel:0
bert/main/encoder/layer_4/mha/query/bias:0
bert/main/encoder/layer_4/mha/key/kernel:0
bert/main/encoder/layer_4/mha/key/bias:0
bert/main/encoder/layer_4/mha/value/kernel:0
bert/main/encoder/layer_4/mha/value/bias:0
bert/main/encoder/layer_4/mha/dense_4/kernel:0
bert/main/encoder/layer_4/mha/dense_4/bias:0
bert/main/encoder/layer_4/attn_layer_norm/gamma:0
bert/main/encoder/layer_4/attn_layer_norm/beta:0
bert/main/encoder/layer_4/intermediate/dense/kernel:0
bert/main/encoder/layer_4/intermediate/dense/bias:0
bert/main/encoder/layer_4/dense/kernel:0
bert/main/encoder/layer_4/dense/bias:0
bert/main/encoder/layer_4/inter_layer_norm/gamma:0
bert/main/encoder/layer_4/inter_layer_norm/beta:0
bert/main/encoder/layer_5/mha/query/kernel:0
bert/main/encoder/layer_5/mha/query/bias:0
bert/main/encoder/layer_5/mha/key/kernel:0
bert/main/encoder/layer_5/mha/key/bias:0
bert/main/encoder/layer_5/mha/value/kernel:0
bert/main/encoder/layer_5/mha/value/bias:0
bert/main/encoder/layer_5/mha/dense_5/kernel:0
bert/main/encoder/layer_5/mha/dense_5/bias:0
bert/main/encoder/layer_5/attn_layer_norm/gamma:0
bert/main/encoder/layer_5/attn_layer_norm/beta:0
bert/main/encoder/layer_5/intermediate/dense/kernel:0
bert/main/encoder/layer_5/intermediate/dense/bias:0
bert/main/encoder/layer_5/dense/kernel:0
bert/main/encoder/layer_5/dense/bias:0
bert/main/encoder/layer_5/inter_layer_norm/gamma:0
bert/main/encoder/layer_5/inter_layer_norm/beta:0
bert/main/encoder/layer_6/mha/query/kernel:0
bert/main/encoder/layer_6/mha/query/bias:0
bert/main/encoder/layer_6/mha/key/kernel:0
bert/main/encoder/layer_6/mha/key/bias:0
bert/main/encoder/layer_6/mha/value/kernel:0
bert/main/encoder/layer_6/mha/value/bias:0
bert/main/encoder/layer_6/mha/dense_6/kernel:0
bert/main/encoder/layer_6/mha/dense_6/bias:0
bert/main/encoder/layer_6/attn_layer_norm/gamma:0
bert/main/encoder/layer_6/attn_layer_norm/beta:0
bert/main/encoder/layer_6/intermediate/dense/kernel:0
bert/main/encoder/layer_6/intermediate/dense/bias:0
bert/main/encoder/layer_6/dense/kernel:0
bert/main/encoder/layer_6/dense/bias:0
bert/main/encoder/layer_6/inter_layer_norm/gamma:0
bert/main/encoder/layer_6/inter_layer_norm/beta:0
bert/main/encoder/layer_7/mha/query/kernel:0
bert/main/encoder/layer_7/mha/query/bias:0
bert/main/encoder/layer_7/mha/key/kernel:0
bert/main/encoder/layer_7/mha/key/bias:0
bert/main/encoder/layer_7/mha/value/kernel:0
bert/main/encoder/layer_7/mha/value/bias:0
bert/main/encoder/layer_7/mha/dense_7/kernel:0
bert/main/encoder/layer_7/mha/dense_7/bias:0
bert/main/encoder/layer_7/attn_layer_norm/gamma:0
bert/main/encoder/layer_7/attn_layer_norm/beta:0
bert/main/encoder/layer_7/intermediate/dense/kernel:0
bert/main/encoder/layer_7/intermediate/dense/bias:0
bert/main/encoder/layer_7/dense/kernel:0
bert/main/encoder/layer_7/dense/bias:0
bert/main/encoder/layer_7/inter_layer_norm/gamma:0
bert/main/encoder/layer_7/inter_layer_norm/beta:0
bert/main/encoder/layer_8/mha/query/kernel:0
bert/main/encoder/layer_8/mha/query/bias:0
bert/main/encoder/layer_8/mha/key/kernel:0
bert/main/encoder/layer_8/mha/key/bias:0
bert/main/encoder/layer_8/mha/value/kernel:0
bert/main/encoder/layer_8/mha/value/bias:0
bert/main/encoder/layer_8/mha/dense_8/kernel:0
bert/main/encoder/layer_8/mha/dense_8/bias:0
bert/main/encoder/layer_8/attn_layer_norm/gamma:0
bert/main/encoder/layer_8/attn_layer_norm/beta:0
bert/main/encoder/layer_8/intermediate/dense/kernel:0
bert/main/encoder/layer_8/intermediate/dense/bias:0
bert/main/encoder/layer_8/dense/kernel:0
bert/main/encoder/layer_8/dense/bias:0
bert/main/encoder/layer_8/inter_layer_norm/gamma:0
bert/main/encoder/layer_8/inter_layer_norm/beta:0
bert/main/encoder/layer_9/mha/query/kernel:0
bert/main/encoder/layer_9/mha/query/bias:0
bert/main/encoder/layer_9/mha/key/kernel:0
bert/main/encoder/layer_9/mha/key/bias:0
bert/main/encoder/layer_9/mha/value/kernel:0
bert/main/encoder/layer_9/mha/value/bias:0
bert/main/encoder/layer_9/mha/dense_9/kernel:0
bert/main/encoder/layer_9/mha/dense_9/bias:0
bert/main/encoder/layer_9/attn_layer_norm/gamma:0
bert/main/encoder/layer_9/attn_layer_norm/beta:0
bert/main/encoder/layer_9/intermediate/dense/kernel:0
bert/main/encoder/layer_9/intermediate/dense/bias:0
bert/main/encoder/layer_9/dense/kernel:0
bert/main/encoder/layer_9/dense/bias:0
bert/main/encoder/layer_9/inter_layer_norm/gamma:0
bert/main/encoder/layer_9/inter_layer_norm/beta:0
bert/main/encoder/layer_10/mha/query/kernel:0
bert/main/encoder/layer_10/mha/query/bias:0
bert/main/encoder/layer_10/mha/key/kernel:0
bert/main/encoder/layer_10/mha/key/bias:0
bert/main/encoder/layer_10/mha/value/kernel:0
bert/main/encoder/layer_10/mha/value/bias:0
bert/main/encoder/layer_10/mha/dense_10/kernel:0
bert/main/encoder/layer_10/mha/dense_10/bias:0
bert/main/encoder/layer_10/attn_layer_norm/gamma:0
bert/main/encoder/layer_10/attn_layer_norm/beta:0
bert/main/encoder/layer_10/intermediate/dense/kernel:0
bert/main/encoder/layer_10/intermediate/dense/bias:0
bert/main/encoder/layer_10/dense/kernel:0
bert/main/encoder/layer_10/dense/bias:0
bert/main/encoder/layer_10/inter_layer_norm/gamma:0
bert/main/encoder/layer_10/inter_layer_norm/beta:0
bert/main/encoder/layer_11/mha/query/kernel:0
bert/main/encoder/layer_11/mha/query/bias:0
bert/main/encoder/layer_11/mha/key/kernel:0
bert/main/encoder/layer_11/mha/key/bias:0
bert/main/encoder/layer_11/mha/value/kernel:0
bert/main/encoder/layer_11/mha/value/bias:0
bert/main/encoder/layer_11/mha/dense_11/kernel:0
bert/main/encoder/layer_11/mha/dense_11/bias:0
bert/main/encoder/layer_11/attn_layer_norm/gamma:0
bert/main/encoder/layer_11/attn_layer_norm/beta:0
bert/main/encoder/layer_11/intermediate/dense/kernel:0
bert/main/encoder/layer_11/intermediate/dense/bias:0
bert/main/encoder/layer_11/dense/kernel:0
bert/main/encoder/layer_11/dense/bias:0
bert/main/encoder/layer_11/inter_layer_norm/gamma:0
bert/main/encoder/layer_11/inter_layer_norm/beta:0
bert/main/pooler/dense/kernel:0
bert/main/pooler/dense/bias:0
bert/mlm/bias:0
bert/mlm/dense/kernel:0
bert/mlm/dense/bias:0
bert/mlm/layer_norm/gamma:0
bert/mlm/layer_norm/beta:0
bert/nsp/dense/kernel:0
bert/nsp/dense/bias:0
```

很显然，变量名不一样。

但是仔细看你会发现，两个网络的结构是一样的，所有的变量都是可以一一对应上的。


## 如何加载预训练模型

至此，两个网络的模型结构都清楚了。那么所谓加载预训练模型，就变成了：**根据你自己模型的变量名，从预训练模型中找到对应的变量，把它的值设置到你自己的模型变量里去！**

这个过程就是纯粹的苦力活了，手动设置对应关系即可。我的代码实现，需要进行以下映射：

```python
    def _build_variables_mapping(self, num_layers):
        # model variable name -> pretrained bert variable name
        m = {
            'bert/main/embedding/weight:0': 'bert/embeddings/word_embeddings',
            'bert/main/embedding/position_embedding/embeddings:0': 'bert/embeddings/position_embeddings',
            'bert/main/embedding/token_type_embedding/embeddings:0': 'bert/embeddings/token_type_embeddings',
            'bert/main/embedding/layer_normalization/gamma:0': 'bert/embeddings/LayerNorm/gamma',
            'bert/main/embedding/layer_normalization/beta:0': 'bert/embeddings/LayerNorm/beta',
        }

        for i in range(num_layers):
            # attention
            for n in ['query', 'key', 'value']:
                k = 'bert/main/encoder/layer_{}/mha/{}/kernel:0'.format(i, n)
                v = 'bert/encoder/layer_{}/attention/self/{}/kernel'.format(i, n)
                m[k] = v
                k = 'bert/main/encoder/layer_{}/mha/{}/bias:0'.format(i, n)
                v = 'bert/encoder/layer_{}/attention/self/{}/bias'.format(i, n)
                m[k] = v

            # dense after attention
            for n in ['kernel', 'bias']:
                k = 'bert/main/encoder/layer_{}/mha/dense/{}:0'.format(i, n)
                v = 'bert/encoder/layer_{}/attention/output/dense/{}'.format(i, n)
                m[k] = v
            # layer norm after attention
            for n in ['gamma', 'beta']:
                k = 'bert/main/encoder/layer_{}/attn_layer_norm/{}:0'.format(i, n)
                v = 'bert/encoder/layer_{}/output/LayerNorm/{}'.format(i, n)
                m[k] = v

            # intermediate
            for n in ['kernel', 'bias']:
                k = 'bert/main/encoder/layer_{}/intermediate/dense/{}:0'.format(i, n)
                v = 'bert/encoder/layer_{}/intermediate/dense/{}'.format(i, n)
                m[k] = v

            # output
            for n in ['kernel', 'bias']:
                k = 'bert/main/encoder/layer_{}/dense/{}:0'.format(i, n)
                v = 'bert/encoder/layer_{}/output/dense/{}'.format(i, n)
                m[k] = v

            # layer norm
            for n in ['gamma', 'beta']:
                k = 'bert/main/encoder/layer_{}/inter_layer_norm/{}:0'.format(i, n)
                v = 'bert/encoder/layer_{}/output/LayerNorm/{}'.format(i, n)
                m[k] = v

        # pooler
        for n in ['kernel', 'bias']:
            k = 'bert/main/pooler/dense/{}:0'.format(n)
            v = 'bert/pooler/dense/{}'.format(n)
            m[k] = v

        # masked lm
        m['bert/mlm/bias:0'] = 'cls/predictions/output_bias'
        for n in ['kernel', 'bias']:
            k = 'bert/mlm/dense/{}:0'.format(n)
            v = 'cls/predictions/transform/dense/{}'.format(n)
            m[k] = v
        for n in ['gamma', 'beta']:
            k = 'bert/mlm/layer_norm/{}:0'.format(n)
            v = 'cls/predictions/transform/LayerNorm/{}'.format(n)
            m[k] = v

        # nsp
        m['bert/nsp/dense/kernel:0'] = 'cls/seq_relationship/output_weights'
        m['bert/nsp/dense/bias:0'] = 'cls/seq_relationship/output_bias'

        return m

```

至此，两个模型的变量名一一对应上了，接下来直接根据名字设置变量值即可。

整个流程的核心代码如下：

```python
    def adapte(self, pretrain_model_dir, **kwargs):
        # 从预训练模型目录，找到对应的文件
        config, ckpt, vocab = self._parse_files(pretrain_model_dir)
        # 根据预训练模型的参数配置，得到我们的模型配置
        model_config = self._map_model_config(config)
        # 根据模型配置，构建出模型
        model = build_pretraining_bert_model(model_config)

        # 加载变量的值，tensorflow提供了API
        loader = self._variable_loader(ckpt)

        weights, values, names = [], [], []
        names_mapping = self._build_variables_mapping(model_config['num_layers'])
        for w in model.trainable_weights:
            if w.name not in names_mapping:
                continue
            names.append(w.name)
            weights.append(w)
            v = loader(names_mapping[w.name])
            if w.name == 'bert/nsp/dense/kernel:0':
                v = v.T
            values.append(v)

        logging.info('weights will be loadded from pretrained checkpoint: \n\t{}'.format('\n\t'.join(names)))

        # (weight, value) weight就是需要设置值的变量名，value就是预训练模型的对应该变量的权重
        mapped_values = zip(weights, values)
        # 使用tensorflow提供的函数设置变量值
        tf.keras.backend.batch_set_value(mapped_values)

        # 返回加载预训练权重的模型
        return model

```


得到模型之后，你可以用来做你想做的任何事情，例如抽取特征啊，得到句子向量表示啊等等。

在我的代码仓库里，实现这个加载预训练的过程可以这样：

```python
from transformers_keras.adapters import BertAdapter

# download the pretrained model and extract it to some path
PRETRAINED_BERT_MODEL = '/path/to/chinese_L-12_H-768_A-12'

adapter = BertAdapter()
model = adapter.adapte(PRETRAINED_BERT_MODEL)

print('model inputs: {}'.format(model.inputs))
print('model outputs: {}'.format(model.outputs))
```

打印结果如下：

```bash
model inputs: [<tf.Tensor 'input_ids:0' shape=(None, 512) dtype=int32>, <tf.Tensor 'segment_ids:0' shape=(None, 512) dtype=int32>, <tf.Tensor 'input_mask:0' shape=(None, 512) dtype=int32>]
model outputs: [<tf.Tensor 'predictions/Identity:0' shape=(512, 21128) dtype=float32>, <tf.Tensor 'relations/Identity:0' shape=(2,) dtype=float32>]
```

以上就是本文所有内容。

欢迎大家`star`我的仓库：[luozhouyang/transformers-keras](https://github.com/luozhouyang/transformers-keras)

