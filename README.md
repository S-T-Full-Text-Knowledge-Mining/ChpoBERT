# ChpoBERT：面向汉语政策文本的预训练模型



## 介绍

政策文本的研究需要自然语言处理工具的支持。

预训练语言模型极大地提高了一般文本的文本挖掘精度。

目前，迫切需要一种专门用于汉语政策全文本自动处理的预训练语言模型。

我们使用国内各个省市政策发布平台上的政策全文本作为训练集，以BERT-base-chinese、Chinese-RoBERTa-wwm-ext两个中文预训练模型为基线，基于MLM和WWM任务构建了ChpoBERT-mlm，ChpoBERT-wwm，ChpoRoBERTa-mlm和ChpoRoBERTa-wwm汉语政策预训练模型。

我们设计了一个政策文本命名实体识别任务，并增加ERNIE模型进行对比，以验证预训练模型的性能。



## 使用方法

### Huggingface Transformers

基于Huggingface Transformers的from_pretraining方法可以在线直接获取ChpoBERT-mlm，ChpoBERT-wwm，ChpoRoBERTa-mlm和ChpoRoBERTa-wwm模型。

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

汉语政策文本预训练数据集涵盖了全国31个省市和经济发展水平前列的20个城市，也包括国家政策方针。实验涉及的训练集来自259个政策发布平台，共包含145043条数据，总字数304434178。数据示例如下。

| 标题                                         | 内容                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| 工信部:钢铁企业负债率有所降低 继续推进去杠杆 | 本报北京9月17日电（记者丁怡婷）“随着供给侧改革的推进，钢铁行业效益得到改善，钢铁企业不断优化资产结构。钢铁行业负债率从今年3月比较高的水平，逐步下降。”工信部有关负责人在16日举行的2017（第六届）中国钢铁技术经济高端论坛上介绍，截至今年7月，大中型钢铁企业资产总额4.9万亿元，同比增长5.65%；负债总额3.42万亿元，同比增长5.23%；资产负债率同比有所降低。资料显示，我国大中型钢铁企业平均资产负债率从2008年突破60%以后逐年上升，在今年3月达到了近年来的高点。钢铁企业“高杠杆”造成了企业较高的财务负担。2016年大中型钢铁企业吨钢财务费用约为141元，在钢铁行业供给侧改革的政策支持下，今年1—7月，大中型钢铁企业吨钢财务费用约为130元，降幅较明显。工信部有关负责人认为，要通过坚定推动去产能、国有企业改革、市场化法治化债转股、企业兼并重组等方式，继续推进钢铁企业去杠杆。 |



### 2.验证数据集

验证数据集来自中华人民共和国国家科学技术部682篇经人工标注的科技政策文本。



## 模型下载

我们所提供的是PyTorch版的模型。



## 下载途径

- 从Huggingface官网下载。

- [ChpoBERT-mlm](https://huggingface.co/KM4STfulltext/ChpoBERT-mlm)

- [ChpoBERT-wwm](https://huggingface.co/KM4STfulltext/ChpoBERT-wwm)

- [ChpoRoBERTa-mlm](https://huggingface.co/KM4STfulltext/ChpoRoBERTa-mlm)
- [ChpoRoBERTa-wwm](https://huggingface.co/KM4STfulltext/ChpoRoBERTa-wwm)



## 验证结果

我们使用四个汉语政策预训练模型、两个基准模型和ERNIE模型在验证集上开展命名实体识别任务，对比预训练模型与基准模型的性能。实验结果如下所示。

| Tag          | PRF  | BERT-base-chinese | Chinese-RoBERTa-wwm-ext | ERNIE  | ChpoBERT-mlm | ChpoRoBERTa-mlm | ChpoBERT-wwm | ChpoRoBERTa-wwm | support |
| ------------ | ---- | ----------------- | ----------------------- | ------ | ------------ | --------------- | ------------ | --------------- | :-----: |
| X1           | P    | 78.07             | 77.96                   | 71.64  | 79.46        | 76.44           | 77.96        | 76.44           |   152   |
|              | R    | 96.05             | 95.39                   | 94.74  | 96.71        | 96.05           | 95.39        | 96.05           |         |
|              | F1   | 86.14             | 85.8                    | 81.59  | 87.24        | 85.13           | 85.80        | 85.13           |         |
| X2           | P    | 91.96             | 86.97                   | 85.54  | 88.51        | 89.91           | 88.51        | 89.87           |   213   |
|              | R    | 96.71             | 97.18                   | 97.18  | 97.65        | 96.24           | 97.65        | 95.77           |         |
|              | F1   | 94.28             | 91.8                    | 90.99  | 92.86        | 92.97           | 92.86        | 92.73           |         |
| X3           | P    | 42.67             | 45.49                   | 45.75  | 40.89        | 45.98           | 46.53        | 44.39           |   142   |
|              | R    | 69.72             | 78.17                   | 79.58  | 71.13        | 72.54           | 80.28        | 66.90           |         |
|              | F1   | 52.94             | 57.51                   | 58.10  | 51.93        | 56.28           | 58.91        | 53.37           |         |
| X4           | P    | 87.88             | 93.94                   | 90.62  | 84.21        | 94.12           | 86.84        | 91.43           |   42    |
|              | R    | 69.05             | 73.18                   | 69.05  | 76.19        | 76.19           | 78.57        | 76.19           |         |
|              | F1   | 77.33             | 82.67                   | 78.38  | 80.00        | 84.21           | 82.50        | 83.12           |         |
| X5           | P    | 88.89             | 100.00                  | 100.00 | 100.00       | 100.00          | 100.00       | 100.00          |   14    |
|              | R    | 57.14             | 57.14                   | 57.14  | 57.14        | 57.14           | 57.14        | 57.14           |         |
|              | F1   | 69.57             | 72.73                   | 72.73  | 72.73        | 72.73           | 72.73        | 72.73           |         |
| macro-avg    | P    | 77.89             | 80.87                   | 78.71  | 78.61        | 81.29           | 79.97        | 80.43           |   563   |
|              | R    | 77.73             | 80.34                   | 79.54  | 79.76        | 79.63           | 81.81        | 78.41           |         |
|              | F1   | 76.05             | 78.10                   | 76.36  | 76.95        | 78.26           | 78.56        | 77.42           |         |
| weighted-avg | P    | 75.40             | 74.92                   | 72.49  | 74.02        | 75.76           | 75.23        | 75.14           |   563   |
|              | R    | 86.68             | 89.16                   | 88.99  | 88.10        | 87.74           | 90.23        | 86.14           |         |
|              | F1   | 79.78             | 80.38                   | 78.76  | 79.56        | 80.44           | 81.12        | 79.54           |         |

