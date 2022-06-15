import os
import json

labels = set()


def preprocess(path, save_path):
  res = []
  with open(path, 'r', encoding='utf-8') as fp:
    data = fp.readlines()
  i = 0
  for d in data:
    d = json.loads(d)
    tmp = {"id":i, 'labels':[]}
    text = d['text']
    tmp['text'] = text
    entities = d['entity_list']
    for j, entity in enumerate(entities):
      entity_type = entity['entity_type']
      ent = entity['entity']
      start = entity['entity_index']['begin']
      end = entity['entity_index']['end']
      tmp['labels'].append(
        ["T{}".format(str(j)), entity_type, int(start), int(end), ent]
      )
      labels.add(entity_type)
      
    res.append(tmp)
  with open(save_path, 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)



if __name__ == "__main__":
  preprocess('train.txt', '../mid_data/train.json')
  preprocess('dev.txt', '../mid_data/dev.json')
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
