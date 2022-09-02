# coding=utf-8
import sys

sys.path.append('..')
import os
import torch
import json
import logging

from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import bert_ner_model
from predict import predict
from cut import cut_sentences_main

logger = logging.getLogger(__name__)
commonUtils.set_logger("predictor.log")

args_path = "../checkpoints/bert_crf_cner/args.json"

with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Predictor:
    def __init__(self):
        args = Dict2Class(**tmp_args)
        args.gpu_ids = "0" if torch.cuda.is_available() else "-1"
        print(args.__dict__)
        other_path = os.path.join('../data/{}'.format(args.data_name), 'mid_data').replace("\\\\", "/")
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k

        model_path = '../checkpoints/{}_{}/model.pt'.format(args.model_name, args.data_name)
        args.bert_dir = '../../model_hub/chinese-bert-wwm-ext/'
        model = bert_ner_model.BertNerModel(args)
        model, device = trainUtils.load_model_and_parallel(model, args.gpu_ids, model_path)
        self.model = model
        self.device = device
        self.id2query = id2query
        self.args = args

    def do_predict(self, raw_text):
        raw_text = cut_sentences_main(raw_text, 510)
        entities = []
        for text in raw_text:
            entity = predict(text, self.model, self.device, self.args, self.id2query)
            entities.append(entity)
        return raw_text, entities

    def merge(self, entities):
        """合并抽取结果"""
        with open(os.path.join('../data/{}'.format(self.args.data_name), 'mid_data/') + 'labels.json', 'r', encoding='utf-8') as fp:
          labels = json.load(fp)
        res = {label:[] for label in labels}
        for tmp in entities:
            for k, entity in tmp.items():
                if entity:
                    for e in entity:
                        if e[0] not in res[k]:
                            res[k].append(e[0])
        # print(res)
        return res


import json
from flask import Flask
from flask import request

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

@app.route("/extraction/", methods=["POST"])
def extract_data():
    if request.method == "POST":
        params = request.get_json()
        text = params["text"]
        if not text:
            return json.dumps({"msg": "输入的文本为空",
                               "ucode": 404,
                               "result": {}
                               }, ensure_ascii=False)
        # 抽取关联信息
        try:
            texts, entities = predictor.do_predict(text)
            res = predictor.merge(entities)
        except Exception as e:
            logger.error(text)
            logger.error(e)
            return json.dumps({"msg": e,
                               "ucode": 500,
                               "result": {}
                               }, ensure_ascii=False)
        return json.dumps({"msg": "success",
                           "ucode": 200,
                           "result": res
                           }, ensure_ascii=False)
    else:
        return json.dumps({"msg": "请求方式为post",
                           "ucode": 500,
                           "result": {}
                           }, ensure_ascii=False)


if __name__ == '__main__':
    predictor = Predictor()
    # raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
    # print(raw_text)
    # print(len(raw_text))
    # texts, entities = predictor.do_predict(raw_text)
    # print(predictor.merge(entities))
    app.run(host="0.0.0.0", port=9277, debug=False, threaded=False)
