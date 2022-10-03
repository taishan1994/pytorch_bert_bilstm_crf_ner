import json
import random
import pandas as pd
from collections import Counter


data = pd.read_csv("./Train_merge.csv", encoding="utf-8")

# ========================
# 列名：文本唯一标识、主体，主体开始，主体结束，评价，评价开始，评价结束，主体类别，情感，文本
"""
Index(['Unnamed: 0', 'id', 'AspectTerms', 'A_start', 'A_end', 'OpinionTerms',
       'O_start', 'O_end', 'Categories', 'Polarities', 'text'],
      dtype='object')
"""
print(data.columns)
# ========================

# ========================
# 统计文本的长度，最大长度为69
text = data['text'].values.tolist()
text = list(set(text))
text_length = list(map(lambda x: len(x), text))
text_length_counter = Counter(text_length)
print(text_length_counter)
text_length_counter = sorted(text_length_counter.items(), key=lambda x: x[0])
print(text_length_counter)
# ========================

# ========================
# 获取主体的类别
cates = data['Categories'].values.tolist()
cates = list(set(cates))
print(cates)

# ========================
# 获取评价类别
polars = data['Polarities'].values.tolist()
polars = list(set(polars))
print(polars)

labels = cates + polars
with open('../mid_data/labels.json', 'w', encoding="utf-8") as fp:
    json.dump(labels, fp, ensure_ascii=False)
# ========================

# ========================
# 制作BIOES标签
i = 1
ent_label = {"O": 0}
for label in labels:
    ent_label['B-{}'.format(label)] = i
    i += 1
    ent_label['I-{}'.format(label)] = i
    i += 1
    ent_label['E-{}'.format(label)] = i
    i += 1
    ent_label['S-{}'.format(label)] = i
    i += 1
with open('../mid_data/nor_ent2id.json', 'w', encoding="utf-8") as fp:
    json.dump(ent_label, fp, ensure_ascii=False)
# ========================

# ========================
# 转换数据为我们所需要的格式
id_set = set()
res = []
tmp = {}
for d in data.iterrows():
    d = d[1]
    did = d[1]
    aspect = d[2]
    a_start = d[3]
    a_end = d[4]
    opinion = d[5]
    o_start = d[6]
    o_end = d[7]
    category = d[8]
    polary = d[9]
    text = d[10]
    # print(did, aspect, a_start, a_end, opinion, o_start, o_end,
    #       category, polary, text)
    if did not in id_set:
        if tmp:
            print(tmp)
            res.append(tmp)
        id_set.add(did)
        tmp = {}
        tmp['id'] = did
        tmp['text'] = text
        tmp['labels'] = []
    try:
        if aspect != "_":
            tmp['labels'].append(["T0", category, int(a_start), int(a_end), aspect])
        if category != "_":
            tmp['labels'].append(["T0", polary, int(o_start), int(o_end), opinion])
    except Exception as e:
        continue

random.seed(123)
random.shuffle(res)

ratio = 0.9
length = len(res)
train_data = res[:int(ratio*length)]
dev_data = res[int(ratio*length):]
with open('../mid_data/train.json', 'w', encoding="utf-8") as fp:
    json.dump(train_data, fp, ensure_ascii=False)
with open('../mid_data/dev.json', 'w', encoding="utf-8") as fp:
    json.dump(dev_data, fp, ensure_ascii=False)
# ========================
