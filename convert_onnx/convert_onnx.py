import sys

sys.path.append('..')
import os
import json
import logging
from pprint import pprint
import torch
import numpy as np
from transformers import BertTokenizer
import time

from bert_ner_model_onnx import BertNerModel
from utils import decodeUtils


class Dict2Obj(dict):
    """让字典可以使用.调用"""

    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value


class ConverttOnnx:
    def __init__(self, args, model, tokenizer, idx2tag):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.idx2tag = idx2tag

    def inference(self, texts):
        self.model.eval()
        with torch.no_grad():
            tokens = [i for i in texts]
            encode_dict = self.tokenizer.encode_plus(text=tokens,
                                                     max_length=self.args.max_seq_len,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True,
                                                     return_attention_mask=True,
                                                     return_tensors="pt")
            token_ids = encode_dict['input_ids']
            attention_masks = encode_dict['attention_mask'].bool()
            token_type_ids = encode_dict['token_type_ids']
            s1 = time.time()
            for i in range(NUM):
                logits = self.model(token_ids, attention_masks, token_type_ids)
            # print(logits)
            e1 = time.time()
            print('原版耗时：', (e1 - s1) / NUM)
            if self.args.use_crf == 'True':
                output = logits
            else:
                output = logits.detach().cpu().numpy()
                output = np.argmax(output, axis=2)
            pred_entities = decodeUtils.bioes_decode(output[0][1:1 + len(texts)], texts, self.idx2tag)
            print(pred_entities)

    def convert(self, save_path):
        self.model.eval()
        inputs = {'token_ids': torch.ones(1, args.max_seq_len, dtype=torch.long),
                  'attention_masks': torch.ones(1, args.max_seq_len, dtype=torch.uint8),
                  'token_type_ids': torch.ones(1, args.max_seq_len, dtype=torch.long)}

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (inputs["token_ids"],
                 inputs["attention_masks"],
                 inputs["token_type_ids"]),
                save_path,
                opset_version=11,
                do_constant_folding=True,
                input_names=["token_ids", "attention_masks", "token_type_ids"],
                output_names=["logits"],
                dynamic_axes={'token_ids': symbolic_names,
                              'attention_masks': symbolic_names,
                              'token_type_ids': symbolic_names,
                              'logits': symbolic_names}
            )

    def onnx_inference(self, ort_session, texts):
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        tokens = [i for i in texts]
        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=args.max_seq_len,
                                            padding="max_length",
                                            truncation="longest_first",
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors="pt")
        token_ids = encode_dict['input_ids']
        attention_masks = torch.tensor(encode_dict['attention_mask'], dtype=torch.uint8)
        token_type_ids = encode_dict['token_type_ids']
        token_ids = to_numpy(token_ids)
        attention_masks = to_numpy(attention_masks)
        token_type_ids = to_numpy(token_type_ids)
        s2 = time.time()
        for i in range(NUM):
            output = ort_session.run(None, {'token_ids': token_ids, "attention_masks": attention_masks,
                                            "token_type_ids": token_type_ids})
        e2 = time.time()
        print('onnx耗时：', (e2 - s2) / NUM)
        output = output[0]
        pred_entities = decodeUtils.bioes_decode(output[0][1:1 + len(texts)], texts, self.idx2tag)
        print(pred_entities)


if __name__ == "__main__":
    # 加载模型、配置及其它的一些
    # ===================================
    ckpt_path = "../checkpoints/bert_crf_c/"
    model_path = os.path.join(ckpt_path, "model.pt")
    args_path = os.path.join(ckpt_path, "args.json")
    with open(args_path, 'r') as fp:
        args = json.load(fp)
        args = Dict2Obj(args)

    args.bert_dir = '../model_hub/chinese-bert-wwm-ext/'
    args.data_dir = '../data/cner/'
    NUM = 10
    model = BertNerModel(args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    pprint('Load ckpt from {}'.format(ckpt_path))

    with open(os.path.join(args.data_dir, "mid_data/nor_ent2id.json"), 'r') as fp:
        labels = json.load(fp)
    id2tag = {v: k for k, v in labels.items()}
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    # ===================================

    # 定义转换器
    # ===================================
    convertOnnx = ConverttOnnx(args, model, tokenizer, id2tag)
    # ===================================

    texts = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"

    # 一般的推理
    # ===================================
    convertOnnx.inference(texts)
    # ===================================

    # 转换成onnx
    # ===================================
    save_path = "model.onnx"
    convertOnnx.convert(save_path)
    # ===================================

    # 转Onnx后的推理
    # ===================================
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(save_path)
    convertOnnx.onnx_inference(ort_session, texts)




