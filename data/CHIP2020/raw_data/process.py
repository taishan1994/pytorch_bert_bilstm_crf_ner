import os
import json

labels = set()


def preprocess(path, save_path):
  res = []
  with open(path, 'r', encoding='utf-8') as fp:
    data = fp.readlines()
  i = 0
  for d in data:
    tmp = {"id":i, 'labels':[]}
    d = d.strip()
    d = d.split('|||')
    text = d[0]
    tmp['text'] = text
    for j,entity in enumerate(d[1:]):
      entity = entity.split('    ')
      if len(entity) == 3:
        tmp['labels'].append(
          ["T{}".format(str(j)), entity[2], int(entity[0]), int(entity[1])+1, text[int(entity[0]):int(entity[1])+1]]
        )
        labels.add(entity[2])
    res.append(tmp)
  with open(save_path, 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)



if __name__ == "__main__":
  preprocess('train_data_all.txt', '../mid_data/train.json')
  preprocess('val_data.txt', '../mid_data/dev.json')
  with open('../mid_data/labels.json', 'w', encoding='utf-8') as fp:
    json.dump(list(labels), fp, ensure_ascii=False)

  ent2id_list = ['O']
  for label in labels:
    ent2id_list.append('B-' + label)
    ent2id_list.append('I-' + label)
    ent2id_list.append('E-' + label)
    ent2id_list.append('S-' + label)
  ent2id_dict = {}
  for i, ent in enumerate(ent2id_list):
    ent2id_dict[ent] = i


  with open('../mid_data/nor_ent2id.json', 'w', encoding='utf-8') as fp:
    json.dump(ent2id_dict, fp, ensure_ascii=False)
