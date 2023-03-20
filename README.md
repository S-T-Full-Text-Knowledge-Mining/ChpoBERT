# ChpoBERT：面向中文政策文本的预训练模型



## 介绍

政策文本的研究需要自然语言处理工具的支持。

预训练语言模型极大地提高了一般文本的文本挖掘精度。

目前，迫切需要一种专门用于中文政策全文本自动处理的预训练语言模型。

我们使用国内各个省市政策发布平台上的政策全文本作为训练集，以BERT-base-chinese、Chinese-RoBERTa-wwm-ext两个中文预训练模型为基线，基于MLM和WWM任务构建了ChpoBERT-mlm，ChpoBERT-wwm，ChpoRoBERTa-mlm和ChpoRoBERTa-wwm中文政策预训练模型。

我们设计了三个下游实验：词汇分词、词性标注和实体识别，并增加ERNIE模型进行对比，以验证预训练模型的性能。



## 使用方法

### Huggingface Transformers

基于Huggingface Transformers的from_pretrained方法可以直接在线获取ChpoBERT-mlm，ChpoBERT-wwm，ChpoRoBERTa-mlm和ChpoRoBERTa-wwm模型。

- ChpoBERT-mlm

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("KM4STfulltext/ChpoBERT-mlm")

model = AutoModel.from_pretrained("KM4STfulltext/ChpoBERT-mlm")
```

- ChpoBERT-wwm

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("KM4STfulltext/ChpoBERT-wwm")

model = AutoModel.from_pretrained("KM4STfulltext/ChpoBERT-wwm")
```

- ChpoRoBERTa-mlm

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("KM4STfulltext/ChpoRoBERTa-mlm")

model = AutoModel.from_pretrained("KM4STfulltext/ChpoRoBERTa-mlm")
```

- ChpoRoBERTa-wwm

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("KM4STfulltext/ChpoRoBERTa-wwm")

model = AutoModel.from_pretrained("KM4STfulltext/ChpoRoBERTa-wwm")
```



## 数据集



### 1.预训练数据集

中文政策文本预训练数据集涵盖了全国30个省市和经济发展水平前列的20个城市，也包括国家政策方针。实验涉及的训练集来自240个政策发布平台，共包含131390条数据，总字数305648206。数据示例如下。

| 标题                               | 内容                                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| 深圳市政府采购评标定标分离管理办法 | 第一章总则第一条为了建立适应高质量发展要求的现代政府采购制度，改进政府采购评审机制和交易机制，形成有效管用、简便易行、有利于实现优质优价采购结果的政府采购竞争机制,助力深圳加快建设中国特色社会主义先行示范区，根据《中华人民共和国政府采购法》《深圳经济特区政府采购条例》《深圳经济特区政府采购条例实施细则》等相关法律法规，结合我市实际情况，制定本办法。第二条评标定标分离（以下简称评定分离）是指在政府集中采购程序中，以公开招标方式执行采购，评审委员会负责对投标文件进行评审、推荐候选中标供应商并出具书面评审报告，由采购人根据评审委员会出具的评审报告从推荐的候选中标供应商中确定中标供应商…… |



### 2.验证数据集

自动分词语料数据来自南京农业大学发布的新时代人民日报分词语料库，经过人工筛选，从中选取与政策相关文本共374篇，总字数为78311字。
自动词性标注语料数据来自课题组拥有的北京大学人民日报经过词性标注的语料，在人工筛选的基础上，共获取了445篇有关政策的语料文本，共计112028字。
实体自动识别语料数据基于所获取的政策文本选取了982篇用以标注实体，共计1918394字。



## 模型下载

我们提供的是PyTorch版的模型。



## 下载途径

- 从Huggingface官网下载。

- [ChpoBERT-mlm](https://huggingface.co/KM4STfulltext/ChpoBERT-mlm)

- [ChpoBERT-wwm](https://huggingface.co/KM4STfulltext/ChpoBERT-wwm)

- [ChpoRoBERTa-mlm](https://huggingface.co/KM4STfulltext/ChpoRoBERTa-mlm)
- [ChpoRoBERTa-wwm](https://huggingface.co/KM4STfulltext/ChpoRoBERTa-wwm)



## 验证结果

我们使用四个中文政策预训练模型、两个基准模型和ERNIE模型在验证集上开展下游任务，对比预训练模型与基准模型的性能。实验结果如下所示。

政策文本自动分词实验结果

| model                   | P          | R          | F          | support |
| ----------------------- | ---------- | ---------- | ---------- | ------- |
| Bert-base-chinese       | 0.9691     | 0.9752     | 0.9721     | 3793    |
| ChpoBert-mlm            | 0.9696     | 0.9757     | 0.9727     | 3793    |
| ChpoBert-wwm            | **0.9727** | **0.9760** | **0.9743** | 3793    |
| Chinese-RoBerta-wwm-ext | 0.9634     | 0.9726     | 0.9680     | 3793    |
| ChpoRoberta-mlm         | 0.9693     | 0.9744     | 0.9719     | 3793    |
| ChpoRoberta-wwm         | 0.9685     | 0.9713     | 0.9699     | 3793    |
| ERNIE                   | 0.9625     | 0.9686     | 0.9656     | 3793    |

政策文本词性标注实验结果

| model                   | weighted avg p | weighted avg r | weighted avg f | support |
| ----------------------- | -------------- | -------------- | -------------- | ------- |
| Bert-base-chinese       | 0.8788         | 0.9033         | 0.8896         | 5221    |
| ChpoBert-mlm            | **0.8902**     | 0.9119         | 0.8999         | 5221    |
| ChpoBert-wwm            | 0.8844         | 0.9060         | 0.8941         | 5221    |
| Chinese-RoBERTa-wwm-ext | 0.8781         | 0.9050         | 0.8903         | 5221    |
| ChpoRoBERTa-mlm         | 0.8898         | **0.9144**     | **0.9012**     | 5221    |
| ChpoRoBERTa-wwm         | 0.8831         | 0.9050         | 0.8930         | 5221    |
| ERNIE                   | 0.8405         | 0.8835         | 0.8595         | 5221    |

政策文本实体识别结果

| models                  | Macro avg  | Weighted avg | support    |            |            |            |      |
| ----------------------- | ---------- | ------------ | ---------- | ---------- | ---------- | ---------- | ---- |
| p                       | r          | f            | p          | r          | f          |            |      |
| Bert-base-chinese       | 0.7590     | 0.8769       | 0.8040     | 0.7672     | 0.8971     | 0.8193     | 763  |
| ChpoBert-mlm            | 0.7478     | 0.8913       | 0.8088     | 0.7573     | 0.9037     | 0.8203     | 763  |
| ChpoBert-wwm            | **0.7741** | **0.9008**   | **0.8243** | **0.7819** | **0.9184** | **0.8379** | 763  |
| Chinese-RoBerta-wwm-ext | 0.7506     | 0.8770       | 0.8013     | 0.7630     | 0.8957     | 0.8170     | 763  |
| ChpoRoberta-mlm         | 0.7686     | 0.8936       | 0.8191     | 0.7805     | 0.9091     | 0.8336     | 763  |
| ChpoRoberta-wwm         | 0.7672     | 0.8966       | 0.8188     | 0.7732     | 0.9118     | 0.8303     | 763  |
| ERNIE                   | 0.6990     | 0.8808       | 0.7702     | 0.7162     | 0.8997     | 0.7894     | 763  |
