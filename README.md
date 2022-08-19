# pytorch_bert_bilstm_crf_ner
延申：
- 百度UIE通用信息抽取：https://github.com/taishan1994/pytorch_uie_ner
- 一种基于globalpointer的命名实体识别：https://github.com/taishan1994/pytorch_GlobalPointer_Ner
- 一种基于TPLinker_plus的命名实体识别：https://github.com/taishan1994/pytorch_TPLinker_Plus_Ner
- 一种one vs rest方法进行命名实体识别：https://github.com/taishan1994/pytorch_OneVersusRest_Ner
- 一种级联Bert用于命名实体识别，解决标签过多问题：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner
- 一种多头选择Bert用于命名实体识别：https://github.com/taishan1994/pytorch_Multi_Head_Selection_Ner
- 中文命名实体识别最新进展：https://github.com/taishan1994/awesome-chinese-ner
- 信息抽取三剑客：实体抽取、关系抽取、事件抽取：https://github.com/taishan1994/chinese_information_extraction
- 一种基于机器阅读理解的命名实体识别：https://github.com/taishan1994/BERT_MRC_NER_chinese
- W2NER：命名实体识别最新sota：https://github.com/taishan1994/W2NER_predict
****
### 温馨提示

- 由于后面重构了些代码，如果使用已经训练好的模型，将模型放置在checkpoints时可能需要修改里面文件夹的名称。
- 新增了转换为onnx并进行推理，具体内容在convert_onnx下，```python convert_onnx.py```，只支持对单条数据的推理。在CPU下，原本推理时间：0.714256477355957s，转换后推理时间：0.4593505859375s。需要安装onnxruntime和onnx库。原本的pytorch-crf不能转换为onnx，这里使用了[here](https://github.com/facebookresearch/pytext/blob/master/pytext/models/crf.py)。目前只测试了bert_crf模型，其余的可根据需要自行调整。
****
由于bert的tokenizer会移除掉空格、换行、制表等字符，因此在utils/common_utils.py里面有一个fine_grade_tokenize函数，该函数是将这些字符用[BLANK]标识，不在vocab.txt的用[INV]标识，因此要先将vocab.txt里面的[unused1]替换为[BLANK], [unused2]替换为[INV]。其实，如果不替换程序也是可以跑的。
****
#### 2022-08-18

- 新增weibo和msra数据，具体运行实例这里不补充，可当练手用。

- 将预测代码提取至predict.py里面，使用时需要注意以下几方面：
	- 修改args_path
	- 修改model_name

#### 2022-08-19

- 新增其它模型的训练结果，目录结构是：

	```
	——project
	————model_hub
	——————chinese-bert-wwm-ext
	————————vocab.txt
	————————pytorch_model.bin
	————————config.json
	——————其它模型路径
	————pytorch_bert_bilstm_crf_ner
	```

- 需要修改的地方是：

	- --bert_dir="../model_hub/chinese-bert-wwm-ext/" 
	- --model_name="bert"
	- 使用electra模型设置model_name="electra"，使用albert模型设置model_name="albert"，使用mengzi模型设置model_name="mengzi"，其余的均可设置model_name="bert"（或自己定义）

| 评价指标：F1                                                 | 模型大小 | PRO  | ORG  | CONT | RACE | NAME | EDU  | LOC  | TITLE | F1     |
| ------------------------------------------------------------ | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ------ |
| [mengzi-bert-base](https://huggingface.co/Langboat/mengzi-bert-base/tree/main) | 196.28M  | 0.90 | 0.91 | 1.00 | 0.93 | 1.00 | 0.96 | 1.00 | 0.90  | 0.9154 |
| [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main) | 392.51M  | 0.90 | 0.92 | 1.00 | 0.93 | 0.99 | 0.96 | 1.00 | 0.91  | 0.9148 |
| [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) | 392.51M  | 0.90 | 0.92 | 1.00 | 0.93 | 1.00 | 0.97 | 1.00 | 0.91  | 0.9233 |
| [chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main) | 392.51M  | 0.90 | 0.92 | 1.00 | 0.93 | 0.99 | 0.97 | 1.00 | 0.90  | 0.9196 |
| [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base/tree/main) | 392.52M  | 0.92 | 0.92 | 1.00 | 0.93 | 1.00 | 0.98 | 1.00 | 0.90  | 0.9203 |
| [chinese-electra-180g-small-discriminator](https://huggingface.co/hfl/chinese-electra-180g-small-discriminator/tree/main) | 47.15M   | 0.74 | 0.88 | 0.99 | 0.12 | 0.97 | 0.81 | 0.00 | 0.87  | 0.8753 |
| [chinese-electra-180g-base-discriminator](https://huggingface.co/hfl/chinese-electra-180g-small-discriminator/tree/main) | 390.17M  | 0.88 | 0.91 | 1.00 | 0.97 | 1.00 | 0.94 | 1.00 | 0.87  | 0.9012 |
| [albert-base-chinese](https://huggingface.co/ckiplab/albert-base-chinese/tree/main) | 38.46M   | 0.00 | 0.68 | 0.95 | 0.00 | 0.62 | 0.53 | 0.00 | 0.71  | 0.6765 |

****

# 补充数据增强实例

在data_augment下的aug.py用于对中文命名实体识别进行数据增强，运行指令：以cner数据集为例

```python
python aug.py --data_name "cner" --text_repeat 2
```

data_name是数据集的名字，text_repeat是每条文本生成文本的数量。在data下需存在data_name的文件夹，先要参考其它数据集生成mid_data下的文件。增强思路：

- 1、先将所有的不同类型的实体都提取出来并存储在/data/cner/aug_data/下。
- 2、将mid_data/train.json中的每一条文本中的实体用**#;#类型#;#**替代，并生成texts.txt在aug_data下。
- 3、遍历texts.txt每一条文本，然后不放回随机从实体库中选择实体替代里面的类型，在和原来train.json里面的数据结合，最终存储在mid_data下的train_aug.json中。
- 4、在preprocess.py里面指定数据集名称，并将use_aug设置为True。接下来的操作与各数据集的运行训练、验证、测试、预测相同。

## 结果

训练、验证、测试和预测运行指令：

```python
!python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cner/" \
--data_name='cner' \
--model_name='bert' \
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
--use_lstm='False' \
--use_crf='True' \
--dropout_prob=0.3 \
--dropout=0.3 \
```

| 评价指标：F1      | PRO  | ORG  | CONT | RACE | NAME | EDU  | LOC  | TITLE | F1     |
| ----------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ------ |
| baseline          | 0.90 | 0.92 | 1.00 | 0.93 | 0.99 | 0.96 | 1.00 | 0.91  | 0.9244 |
| baseline+数据增强 | 0.92 | 0.93 | 1.00 | 0.97 | 1.00 | 0.97 | 1.00 | 0.91  | 0.9293 |

除了数据量不一样，其余的参数均设置为一致。

# 补充分词实例

数据来源：链接: https://pan.baidu.com/s/1gvtqpjz05BglTy597AqbKQ?pwd=xuvp 提取码: xuvp 。具体实验过程参考其它数据集说明。

```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/sighan2005/" \
--data_name='sighan2005' \
--model_name='bert' \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=5 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=512 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=16 \
--train_epochs=3 \
--eval_batch_size=16 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='False' \
--use_crf='True' \
--dropout_prob=0.3 \
--dropout=0.3 

precision:0.9667 recall:0.9549 micro_f1:0.9608
          precision    recall  f1-score   support

    word       0.97      0.95      0.96    104371

micro-f1       0.97      0.95      0.96    104371

在１９９８年来临之际，我十分高兴地通过中央人民广播电台、中国国际广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞、澳门和台湾同胞、海外侨胞，向世界各国的朋友们，致以诚挚的问候和良好的祝愿！
Load ckpt from ./checkpoints/bert_crf_sighan2005/model.pt
Use single gpu in: ['0']
{'word': [('在', 0), ('１９９８年', 1), ('来临', 6), ('之际', 8), ('，', 10), ('我', 11), ('十分', 12), ('高兴', 14), ('地', 16), ('通过', 17), ('中央', 19), ('人民', 21), ('广播', 23), ('电台', 25), ('、', 27), ('中国', 28), ('国际', 30), ('广播', 32), ('电台', 34), ('和', 36), ('中央', 37), ('电视台', 39), ('，', 42), ('向', 43), ('全国', 44), ('各族', 46), ('人民', 48), ('，', 50), ('向', 51), ('香港', 52), ('特别', 54), ('行政区', 56), ('同胞', 59), ('、', 61), ('澳门', 62), ('和', 64), ('台湾', 65), ('同胞', 67), ('、', 69), ('海外', 70), ('侨胞', 72), ('，', 74), ('向', 75), ('世界各国', 76), ('的', 80), ('朋友', 81), ('们', 83), ('，', 84), ('致以', 85), ('诚挚', 87), ('的', 89), ('问候', 90), ('和', 92), ('良好', 93), ('的', 95), ('祝愿', 96), ('！', 98)]}
```

# 补充商品标题要素抽取实例

数据来源：[商品标题](https://www.heywhale.com/mw/dataset/6241349d93e61600170895e5/file)，就一个train.txt，初始格式为BIO。具体实验过程参考其它数据集说明。这里并没有运行完3个epoch，在720步手动终止了。类别数据进行了脱敏，要知道每类是什么意思，只有自己根据数据自己总结了=，=。
```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/attr/" \
--data_name='attr' \
--model_name='bert' \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=209 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=64 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=64 \
--train_epochs=3 \
--eval_batch_size=64 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='False' \
--use_crf='True' \
--dropout_prob=0.1 \
--dropout=0.1 \

precision:0.7420 recall:0.7677 micro_f1:0.7546
          precision    recall  f1-score   support

      17       0.00      0.00      0.00         4
      24       0.00      0.00      0.00         2
      35       0.00      0.00      0.00         0
      19       0.00      0.00      0.00        19
      47       0.57      0.01      0.03       282
      30       0.26      0.09      0.13       111
      12       0.75      0.82      0.78      2460
      44       0.00      0.00      0.00         8
      49       0.32      0.33      0.33       266
      31       0.44      0.23      0.30       169
       1       0.82      0.89      0.85      5048
      20       0.52      0.11      0.18       120
      26       0.00      0.00      0.00         0
      39       0.39      0.30      0.34      1059
      36       0.42      0.53      0.47       736
       5       0.75      0.74      0.74      7982
      11       0.72      0.81      0.76     12250
       6       0.58      0.79      0.67       303
      18       0.73      0.77      0.75     11123
      37       0.74      0.73      0.73      3080
      42       0.00      0.00      0.00         4
      46       0.00      0.00      0.00         7
      33       0.00      0.00      0.00         4
      23       0.00      0.00      0.00         4
      15       0.62      0.58      0.60       146
      28       0.00      0.00      0.00         8
       9       0.50      0.61      0.55      2532
      51       0.00      0.00      0.00         7
      34       0.20      0.06      0.09        54
       4       0.81      0.85      0.83     33645
      14       0.87      0.89      0.88      4553
      13       0.70      0.72      0.71     12992
      32       0.00      0.00      0.00         8
      38       0.60      0.68      0.64      6788
      40       0.75      0.61      0.67      6588
      53       0.00      0.00      0.00         0
      43       0.00      0.00      0.00        13
      22       0.38      0.32      0.35      1770
      48       0.00      0.00      0.00        42
       2       0.26      0.15      0.19       598
      41       0.52      0.11      0.18       108
      29       0.75      0.77      0.76       841
      52       0.00      0.00      0.00        27
      54       0.69      0.65      0.67      1221
       3       0.52      0.61      0.56      1840
       7       0.83      0.92      0.87      4921
      10       0.49      0.46      0.48      1650
      21       0.24      0.26      0.25       120
      25       0.00      0.00      0.00         3
      16       0.90      0.92      0.91      4604
      50       0.56      0.38      0.46        91
       8       0.86      0.90      0.88      3515

micro-f1       0.74      0.77      0.75    133726

荣耀V9Play支架手机壳honorv9paly手机套新品情女款硅胶防摔壳
Load ckpt from ./checkpoints/bert_crf_attr/model.pt
Use single gpu in: ['0']
{'38': [('荣耀V9Play', 0), ('honorv9paly', 13)], '22': [('支架', 8)], '4': [('手机壳', 10), ('手机套', 24), ('防摔壳', 34)], '14': [('新品', 27)], '8': [('情女款', 29)], '12': [('硅胶', 32)]}
```

# 补充地址要素抽取实例
数据集来源是：[CCKS2021中文NLP地址要素解析](https://tianchi.aliyun.com/competition/entrance/531900/information)，报名后可下载数据，这里不提供。具体实验过程参考其它数据集说明。
```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/addr/" \
--data_name='addr' \
--model_name='bert' \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=69 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=64 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=64 \
--train_epochs=3 \
--eval_batch_size=64 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='False' \
--use_crf='True' \
--dropout_prob=0.1 \
--dropout=0.1 \

precision:0.9233 recall:0.9021 micro_f1:0.9125
               precision    recall  f1-score   support

     district       0.96      0.93      0.94      1444
village_group       0.91      0.87      0.89        47
       roadno       0.98      0.98      0.98       815
          poi       0.77      0.85      0.81      1279
       subpoi       0.82      0.65      0.73       459
    community       0.81      0.70      0.75       373
     distance       1.00      1.00      1.00         6
         city       0.99      0.94      0.96      1244
         road       0.94      0.95      0.95      1244
         prov       0.99      0.97      0.98       994
      floorno       0.97      0.94      0.95       211
       assist       0.82      0.88      0.85        64
       cellno       0.99      0.98      0.98       123
         town       0.95      0.87      0.91       924
      devzone       0.82      0.82      0.82       222
      houseno       0.97      0.96      0.97       496
 intersection       0.93      0.65      0.76        20

     micro-f1       0.92      0.90      0.91      9965
    
浙江省嘉兴市平湖市钟埭街道新兴六路法帝亚洁具厂区内万杰洁具
Load ckpt from ./checkpoints/bert_crf_addr/model.pt
Use single gpu in: ['0']
{'prov': [('浙江省', 0)], 'city': [('嘉兴市', 3)], 'district': [('平湖市', 6)], 'town': [('钟埭街道', 9)], 'road': [('新兴六路', 13)], 'poi': [('法帝亚洁具厂区', 17), ('万杰洁具', 25)]}
```

# 补充CLUE实例

具体流程和医疗的类似，原始数据可以从这里下载：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/cluener_public
```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/CLUE/" \
--data_name='clue' \
--model_name='bert' \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=41 \
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
--use_lstm='False' \
--use_crf='True' \
--dropout_prob=0.3 \
--dropout=0.3 \

precision:0.7802 recall:0.8176 micro_f1:0.7984
              precision    recall  f1-score   support

    position       0.77      0.82      0.80       425
       movie       0.88      0.77      0.82       150
        name       0.84      0.90      0.87       451
        book       0.86      0.81      0.83       152
     address       0.65      0.68      0.66       364
organization       0.81      0.81      0.81       344
       scene       0.73      0.76      0.74       199
  government       0.77      0.87      0.82       244
        game       0.76      0.90      0.82       287
     company       0.80      0.81      0.81       366

    micro-f1       0.78      0.82      0.80      2982

彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，
Load ckpt from ./checkpoints/bert_crf/model.pt
Use single gpu in: ['0']
{'name': [('彭小军', 0)], 'address': [('台湾', 15)]}
```

****
# 补充医疗实例
1、在data/CHIP2020/raw_data下是原始数据，使用process.py处理raw_data以获取mid_data下的数据。原始数据可以去这里下载：https://github.com/zhangzhiyi0108/CHIP2020_Entity<br>
2、修改preprocess.py里面为自己定义的数据集，并指定数据地址及最大长度，稍后的自定义参数需要保持和这里的一致。<br>
3、修改main.py里面为自己定义的数据集及相关参数。<br>
4、修改main.sh里面运行指令的相关参数，最后运行即可。<br>
5、基于bert_crf训练好的模型可以去这里下载：链接：https://pan.baidu.com/s/1if6G00ERfXSWfe_h23hgDg?pwd=2s3e 
提取码：2s3e
```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/CHIP2020/" \
--data_name='chip' \
--model_name='bert' \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=37 \
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
--use_lstm='False' \
--use_crf='True' \
--dropout_prob=0.3 \
--dropout=0.3 \

Load ckpt from ./checkpoints/bert_crf/model.pt
Use single gpu in: ['0']
precision:0.6477 recall:0.6530 micro_f1:0.6503
          precision    recall  f1-score   support

     equ       0.57      0.57      0.57       238
     sym       0.59      0.45      0.51      4130
     pro       0.60      0.68      0.64      2057
     bod       0.63      0.66      0.64      5883
     dis       0.71      0.78      0.74      4935
     dru       0.77      0.86      0.81      1440
     mic       0.73      0.82      0.77       584
     dep       0.59      0.53      0.56       110
     ite       0.47      0.40      0.43       923

micro-f1       0.65      0.65      0.65     20300

大动脉转换手术要求左心室流出道大小及肺动脉瓣的功能正常，但动力性左心室流出道梗阻并非大动脉转换术的禁忌证。
Load ckpt from ./checkpoints/bert_crf/model.pt
Use single gpu in: ['0']
{'pro': [('大动脉转换手术', 0), ('大动脉转换术', 42)], 'bod': [('左心室流出道', 9), ('肺动脉瓣', 18)], 'dis': [('动力性左心室流出道梗阻', 29)]}
```
****
# 最初说明
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
transformers==4.5.0
```

# 运行命令
```python
python main.py \
--bert_dir="../model_hub/bert-base-chinese/" \
--data_dir="./data/cner/" \
--data_name='cner' \
--model_name='bert' \
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
--use_lstm='False' \
--use_crf='True' \
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

