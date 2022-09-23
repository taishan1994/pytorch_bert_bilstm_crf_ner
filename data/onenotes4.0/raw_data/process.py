import pandas as pd
import os
import re
import json


def preprocess(input_path, save_path, mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(save_path, mode + ".json")
    labels = set()
    result = []
    tmp = {}
    tmp['id'] = 0
    tmp['text'] = ''
    tmp['labels'] = []
    # =======先找出句子和句子中的所有实体和类型=======
    texts = []
    entities = []
    words = []
    entity_tmp = []
    entities_tmp = []
    data = pd.read_csv(input_path)
    for i,d in enumerate(data.iterrows()):
        d = d[1]
        word = d[0]
        label = d[1]
        if str(word).lower() != "nan":
            words.append(word)
            if "B-" in label:
                entity_tmp.append(word)
            elif "M-" in label:
                entity_tmp.append(word)
            elif "E-" in label:
                entity_tmp.append(word)
                if ("".join(entity_tmp), label.split("-")[-1]) not in entities_tmp:
                    entities_tmp.append(("".join(entity_tmp), label.split("-")[-1]))
                labels.add(label.split("-")[-1])
                entity_tmp = []

            if "S-" in label:
                entity_tmp.append(word)
                if ("".join(entity_tmp), label.split("-")[-1]) not in entities_tmp:
                    entities_tmp.append(("".join(entity_tmp), label.split("-")[-1]))
                entity_tmp = []
                labels.add(label.split("-")[-1])
        else:
            texts.append("".join(words))
            entities.append(entities_tmp)
            words = []
            entities_tmp = []

        # for text,entity in zip(texts, entities):
        #     print(text, entity)
        # print(labels)
    # ==========================================
    # =======找出句子中实体的位置=======
    i = 0
    for text,entity in zip(texts, entities):

        if entity:
            ltmp = []
            for ent,type in entity:
                for span in re.finditer(ent, text):
                    start = span.start()
                    end = span.end()
                    ltmp.append((type, start, end, ent))
                    # print(ltmp)
            ltmp = sorted(ltmp, key=lambda x:(x[1],x[2]))
            tmp['id'] = i
            tmp['text'] = text
            for j in range(len(ltmp)):
                tmp['labels'].append(["T{}".format(str(j)), ltmp[j][0], ltmp[j][1], ltmp[j][2], ltmp[j][3]])
        else:
            tmp['id'] = i
            tmp['text'] = text
            tmp['labels'] = []
        result.append(tmp)
        # print(i, text, entity, tmp)
        tmp = {}
        tmp['id'] = 0
        tmp['text'] = ''
        tmp['labels'] = []
        i += 1

    with open(data_path,'w', encoding='utf-8') as fp:
        fp.write(json.dumps(result, ensure_ascii=False))

    if mode == "train":
        label_path = os.path.join(save_path, "labels.json")
        with open(label_path, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(list(labels), ensure_ascii=False))


preprocess("char_ner_train.csv", '../mid_data', "train")
labels_path = os.path.join('../mid_data/labels.json')
with open(labels_path, 'r') as fp:
    labels = json.load(fp)

tmp_labels = []
tmp_labels.append('O')
for label in labels:
    tmp_labels.append('B-' + label)
    tmp_labels.append('I-' + label)
    tmp_labels.append('E-' + label)
    tmp_labels.append('S-' + label)

label2id = {}
for k,v in enumerate(tmp_labels):
    label2id[v] = k
path  = '../mid_data/'
if not os.path.exists(path):
    os.makedirs(path)
with open(os.path.join(path, "nor_ent2id.json"),'w') as fp:
    fp.write(json.dumps(label2id, ensure_ascii=False))
