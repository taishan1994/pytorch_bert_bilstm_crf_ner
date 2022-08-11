import copy
import glob
import json
import os
import random
import re
from pprint import pprint

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='c',
                            help='数据集名字')
parser.add_argument('--text_repeat', type=int, default=2,
                            help='增强的数目')

args = parser.parse_args()
data_dir = "../data/{}".format(args.data_name)
text_repeat = args.text_repeat
if not os.path.exists(data_dir):
    raise Exception("请确认数据集是否存在")

train_file = os.path.join(data_dir, "mid_data/train.json")
labels_file = os.path.join(data_dir, "mid_data/labels.json")

output_dir = os.path.join(data_dir, "aug_data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_data():
    # ["PRO", "ORG", "CONT", "RACE", "NAME", "EDU", "LOC", "TITLE"]

    """获取基本的数据"""
    with open(train_file, "r", encoding="utf-8") as fp:
        data = fp.read()

    with open(labels_file, "r", encoding="utf-8") as fp:
        labels = json.loads(fp.read())

    entities = {k:[] for k in labels}

    texts = []

    data = json.loads(data)
    for d in data:
        text = d['text']
        labels = d['labels']
        for label in labels:
            text = text.replace(label[4], "#;#{}#;#".format(label[1]))
            entities[label[1]].append(label[4])
        texts.append(text)

    for k,v in entities.items():
        with open(output_dir + "/" + k + ".txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(list(set(v))))

    with open(output_dir + "/texts.txt", 'w', encoding="utf-8") as fp:
        fp.write("\n".join(texts))

def aug_by_template(text_repeat=2):
    """基于模板的增强
    text_repeat:每条文本重复的次数
    """
    with open(output_dir + "/texts.txt", 'r', encoding="utf-8") as fp:
        texts = fp.read().strip().split('\n')

    with open(labels_file, "r", encoding="utf-8") as fp:
        labels = json.loads(fp.read())

    entities = {}
    for ent_txt in glob.glob(output_dir + "/*.txt"):
        if "texts.txt" in ent_txt:
            continue
        with open(ent_txt, 'r', encoding="utf-8") as fp:
            label = fp.read().strip().split("\n")
            ent_txt = ent_txt.replace("\\", "/")

            label_name = ent_txt.split("/")[-1].split(".")[0]
            entities[label_name] = label

    entities_copy = copy.deepcopy(entities)

    with open(train_file, "r", encoding="utf-8") as fp:
        ori_data = json.loads(fp.read())

    res = []
    text_id = ori_data[-1]['id'] + 1
    for text in tqdm(texts, ncols=100):
        text = text.split("#;#")
        text_tmp = []
        labels_tmp = []
        for i in range(text_repeat):
            ent_id = 0
            for t in text:
                if t == "":
                    continue
                if t in entities:
                    # 不放回抽样，为了维持实体的多样性
                    if not entities[t]:
                        entities[t] = copy.deepcopy(entities_copy[t])
                    ent = random.choice(entities[t])
                    entities[t].remove(ent)
                    length = len("".join(text_tmp))
                    text_tmp.append(ent)
                    labels_tmp.append(("T{}".format(ent_id), t, length, length + len(ent), ent))
                    ent_id += 1
                else:
                    text_tmp.append(t)
            tmp = {
                "id": text_id,
                "text": "".join(text_tmp),
                "labels": labels_tmp
            }
            text_id += 1
            text_tmp = []
            labels_tmp = []
            res.append(tmp)
    # 加上原始的
    res = ori_data + res

    with open(data_dir + "/mid_data/train_aug.json", "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False)


if __name__ == '__main__':
    # 1、第一步：获取基本数据
    get_data()
    # 2、第二步：进行模板类数据增强
    aug_by_template(text_repeat=text_repeat)