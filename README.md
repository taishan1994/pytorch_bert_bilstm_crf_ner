# pytorch_bert_bilstm_crf_ner

基于pytorch的bert_bilstm_crf中文命名实体识别<br>
要预先下载好预训练的bert模型，放在和该项目同级下的model_hub文件夹下，即：<br>
model_hub/bert-base-chinese/<br>
相关下载地址：<a href="https://huggingface.co/bert-base-chinese/tree/main=">bert-base-chinese</a><br>
需要的是vocab.txt、config.json、pytorch_model.bin<br>
你也可以使用我已经训练好的模型，将其放在checkpoints下：<br>
链接：https://pan.baidu.com/s/1yIGnQ9I_4HAfQSqHod-hMQ <br>
提取码：4j47<br>
里面有四种模型对应的model.pt<br>

# 目录结构
--checkpoints：模型保存的位置<br>
--data：数据位置<br>
--|--cner：数据集名称<br>
--|--|--raw_data：原始数据存储位置，里面有个process.py用于转换文本+标签json<br>
--|--|--mid_data：保存处理之后的json文件，标签等；<br>
--|--|--final_data：存储处理好之后可用的pickle文件<br>
--logs：日志存储位置<br>
--utils：辅助函数存储位置，包含了解码、评价指标、设置随机种子、设置日志等<br>
--config.py：配置文件<br>
--dataset.py：数据转换为pytorch的DataSet<br>
--main.py：主运行程序<br>
--main.sh：运行命令<br>
--bert_base_model.py：Bert模型<br>
--bert_ner_modelpy：利用Bert进行Ner的模型<br>
--preprocess.py：预处理，主要是处理数据然后转换成DataSet<br>

# 依赖
```python
python==3.6
pytorch==1.6.0
pytorch-crf==0.7.2
```

# 运行命令
```python
python main.py \
--bert_dir="../model_hub/bert-base-chinese/" \
--data_dir="./data/cner/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=33 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=3 \
--eval_batch_size=32 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='True' \
--use_crf='False' \
--dropout_prob=0.3 \
--dropout=0.3 \
```
我们可以通过控制--use_lstm和--use_crf来切换使用bilstm或crf。

# 结果
# 训练、验证、测试和预测
由于忘记保存其它测试和预测了，这里就只展示bert的。
```python
2021-08-05 16:19:12,787 - INFO - main.py - train - 52 - 【train】 epoch:2 359/360 loss:0.0398
2021-08-05 16:19:14,717 - INFO - main.py - train - 56 - [eval] loss:1.8444 precision=0.9484 recall=0.8732 f1_score=0.9093
2021-08-05 16:32:20,751 - INFO - main.py - test - 130 -           
             precision    recall  f1-score   support

     PRO       0.86      0.63      0.73        19
     ORG       0.94      0.91      0.92       543
    CONT       1.00      1.00      1.00        33
    RACE       1.00      0.93      0.97        15
    NAME       0.99      0.93      0.96       110
     EDU       0.98      0.94      0.96       109
     LOC       0.00      0.00      0.00         2
   TITLE       0.95      0.84      0.89       770

micro-f1       0.95      0.88      0.91      1601

2021-08-05 16:32:20,752 - INFO - main.py - <module> - 218 - 虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
2021-08-05 16:32:22,892 - INFO - trainUtils.py - load_model_and_parallel - 96 - Load ckpt from ./checkpoints/bert/model.pt
2021-08-05 16:32:23,205 - INFO - trainUtils.py - load_model_and_parallel - 106 - Use single gpu in: ['0']
2021-08-05 16:32:23,239 - INFO - main.py - predict - 156 - {'NAME': [('虞兔良', 0)], 'RACE': [('汉族', 17)], 'CONT': [('中国国籍', 20)], 'TITLE': [('中共党员', 40), ('经济师', 49)], 'EDU': [('MBA', 45)]}
```
## 验证集上对比
| models | loss | precision | recall | f1_score |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
|bert|1.8444 |0.9484 |0.8732 |0.9093 |
|bert_bilstm|2.0856 |0.9540 |0.8670 |0.9084 |
|bert_crf|26.9665 |0.9385 |0.8957 |0.9166 |
|bert_bilstm_crf|30.8463 |0.9382 |0.8919 |0.9145 |

以上训练的都是3个epoch。

# 补充
[中文命名实体识别最新进展](https://github.com/taishan1994/awesome-chinese-ner)
[信息抽取三剑客：实体抽取、关系抽取、事件抽取](https://github.com/taishan1994/chinese_information_extraction) <br>
[基于机器阅读理解的命名实体识别](https://github.com/taishan1994/BERT_MRC_NER_chinese)<br>
[W2NER：命名实体识别最新sota](https://github.com/taishan1994/W2NER_predict)<br>
