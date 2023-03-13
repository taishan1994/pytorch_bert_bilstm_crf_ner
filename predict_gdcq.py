import os
import re
import numpy as np
import torch
from torch import Tensor
import json
from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import bert_ner_model
from transformers import BertTokenizer


def decode(decode_tokens, raw_text, id2ent):
    predict_entities = []
    if isinstance(decode_tokens, Tensor):
        decode_tokens = decode_tokens.numpy().tolist()
    index_ = 0

    while index_ < len(decode_tokens):
        if decode_tokens[index_] == 0:
            token_label = id2ent[1].split('-')
        else:
            token_label = id2ent[decode_tokens[index_]].split('-')
        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = raw_text[index_]

            predict_entities.append((tmp_ent, index_, token_type))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                if decode_tokens[index_] == 0:
                    temp_token_label = id2ent[1].split('-')
                else:
                    temp_token_label = id2ent[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    tmp_ent = raw_text[start_index: end_index + 1]

                    predict_entities.append((tmp_ent, start_index, token_type))
                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities


def predict(raw_text, model, device, args, id2query):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        tokenizer = BertTokenizer(
            os.path.join(args.bert_dir, 'vocab.txt'))
        # tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
        tokens = [i for i in raw_text]
        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=args.max_seq_len,
                                            padding='max_length',
                                            truncation='longest_first',
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)
        # tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).long().unsqueeze(0).to(device)
        # attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0).to(device)
        try:
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0).to(device)
        except Exception as e:
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).long().unsqueeze(0).to(device)
        token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(device)
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
        if args.use_crf == 'True':
            output = logits
        else:
            output = logits.detach().cpu().numpy()
            output = np.argmax(output, axis=2)

        pred_entities = decode(output[0][1:1 + len(tokens)], "".join(tokens), id2query)
        # print(pred_entities)
        return pred_entities


import re
tmp = ['正面', '中性', '负面']
def post_process(entities, text):
  """
  后处理操作：如果主体和评价在同一段文本里面，且主体在前，则可合并主体和评价
  在此基础上：
    如存在多个主体对一个评价，则多对一
    如存在一个主体和多个评价，则一对多
  """
  if len(entities) <= 1:
    return entities, []
  entities_res = []
  relation_res = []
  tmp_entities_res = []
  for i in range(len(entities) - 1):
    # print(entities[i], entities[i+1])
    if entities[i][-1] not in tmp and entities[i+1][-1] in tmp:
      left_end = entities[i][1] + len(entities[i][0])
      right_start = entities[i+1][1]
      tmp_text = text[left_end:right_start]
      if sum([1 if x in tmp_text else 0 for x in ['，',',','。','！','!','？','?']]) == 0:
        relation_res.append((entities[i][0]+entities[i+1][0], entities[i+1][-1]))
        tmp_entities_res.append(entities[i])
        tmp_entities_res.append(entities[i+1])
  entities_res = [i for i in entities if i not in tmp_entities_res]
  return entities_res, relation_res




if __name__ == "__main__":
  args_path = "checkpoints/bert_crf_gdcq/args.json"

  with open(args_path, "r", encoding="utf-8") as fp:
      tmp_args = json.load(fp)

  class Dict2Class:
      def __init__(self, **entries):
          self.__dict__.update(entries)

  args = Dict2Class(**tmp_args)
  args.gpu_ids = "0" if torch.cuda.is_available() else "-1"
  print(args.__dict__)

  other_path = os.path.join(args.data_dir, 'mid_data')
  ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
  query2id = {}
  id2query = {}
  for k, v in ent2id_dict.items():
      query2id[k] = v
      id2query[v] = k

  raw_text = "***的化妆品还是不错的，值得购买，性价比很高的活动就参加了！！！"
  raw_text = "挺好用的，以前皮肤很容易过敏，用了好多种都过敏，用了这套后就慢慢不过敏了，用完继续"
  raw_text = "多次购买了，效果不错哦，价格便宜"
  print(raw_text)
  model_name = args.model_name
  model_path = './checkpoints/{}_{}/model.pt'.format(model_name, args.data_name)
  if args.model_name.split('_')[0] not in ['bilstm', 'crf', "idcnn"]:
    model = bert_ner_model.BertNerModel(args)
  else:
    model = bert_ner_model.BilstmNerModel(args)
  # print(model)
  model, device = trainUtils.load_model_and_parallel(model, args.gpu_ids, model_path)
  entities = predict(raw_text, model, device, args, id2query)
  print("实体识别结果：", entities)
  entities_res, relation_res = post_process(entities, raw_text)
  print("未进行关联的实体：", entities_res)
  print("关系合并：", relation_res)
