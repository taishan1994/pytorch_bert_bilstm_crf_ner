import os
import json
import logging
from transformers import BertTokenizer
from utils import cutSentences, commonUtils
import config

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels


class NerProcessor:
    def __init__(self, cut_sent=True, cut_sent_len=256):
        self.cut_sent = cut_sent
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            # print(i,item)
            text = item['text']
            if self.cut_sent:
                sentences = cutSentences.cut_sent_for_bert(text, self.cut_sent_len)
                start_index = 0

                for sent in sentences:
                    labels = cutSentences.refactor_labels(sent, item['labels'], start_index)

                    start_index += len(sent)

                    examples.append(InputExample(set_type=set_type,
                                                 text=sent,
                                                 labels=labels))
            else:
                labels = item['labels']
                if len(labels) != 0:
                    labels = [(label[1],label[4],label[2]) for label in labels]
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             labels=labels))
        return examples


def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, ent2id, labels):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    # 文本元组
    callback_info = (raw_text,)
    # 标签字典
    callback_labels = {x: [] for x in labels}
    # _label:实体类别 实体名 实体起始位置
    for _label in entities:
        # print(_label)
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info += (callback_labels,)
    # 序列标注任务 BERT 分词器可能会导致标注偏
    # tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
    tokens = [i for i in raw_text]

    assert len(tokens) == len(raw_text)

    label_ids = None

    # information for dev callback
    # ========================
    label_ids = [0] * len(tokens)

    # tag labels  ent ex. (T1, DRUG_DOSAGE, 447, 450, 小蜜丸)
    for ent in entities:
        # ent: ('PER', '陈元', 0)
        ent_type = ent[0] # 类别

        ent_start = ent[-1] # 起始位置
        ent_end = ent_start + len(ent[1]) - 1

        if ent_start == ent_end:
            label_ids[ent_start] = ent2id['S-' + ent_type]
        else:
            label_ids[ent_start] = ent2id['B-' + ent_type]
            label_ids[ent_end] = ent2id['E-' + ent_type]
            for i in range(ent_start + 1, ent_end):
                label_ids[i] = ent2id['I-' + ent_type]


    if len(label_ids) > max_seq_len - 2:
        label_ids = label_ids[:max_seq_len - 2]

    label_ids = [0] + label_ids + [0]

    # pad
    if len(label_ids) < max_seq_len:
        pad_length = max_seq_len - len(label_ids)
        label_ids = label_ids + [0] * pad_length  # CLS SEP PAD label都为O

    assert len(label_ids) == max_seq_len, f'{len(label_ids)}'

    # ========================
    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation='longest_first',
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        print(tokenizer.decode(token_ids[:len(raw_text)+2]))
        logger.info(f'text: {str(" ".join(tokens))}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"labels: {label_ids}")
        logger.info('length: ' + str(len(token_ids)))
        # for word, token, attn, label in zip(tokens, token_ids, attention_masks, label_ids):
        #   print(word + ' ' + str(token) + ' ' + str(attn) + ' ' + str(label))
    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
    )

    return feature, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id, labels):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        # 有可能text为空，过滤掉
        if not example.text:
          continue
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            ent2id=ent2id,
            tokenizer=tokenizer,
            labels = labels,
        )
        if feature is None:
            continue
        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_data(processor, raw_data_path, json_file, mode, ent2id, labels, args):
    raw_examples = processor.read_json(os.path.join(raw_data_path, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, ent2id, labels)
    save_path = os.path.join(args.data_dir, 'final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data

def save_file(filename, data ,id2ent):
    features, callback_info = data
    file = open(filename,'w',encoding='utf-8')
    for feature,tmp_callback in zip(features, callback_info):
        text, gt_entities = tmp_callback
        for word, label in zip(text, feature.labels[1:len(text)+1]):
            file.write(word + ' ' + id2ent[label] + '\n')
        file.write('\n')
    file.close()


if __name__ == '__main__':

    dataset = "cner"
    args = config.Args().get_parser()
    args.bert_dir = '../model_hub/chinese-bert-wwm-ext/'
    commonUtils.set_logger(os.path.join(args.log_dir, 'preprocess.log'))

    use_aug = False

    if dataset == "cner":
        args.data_dir = './data/cner'
        args.max_seq_len = 150

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        if use_aug:
            train_data = get_data(processor, mid_data_path, "train_aug.json", "train", ent2id, labels, args)
        else:
            train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path,"cner_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "dev.json", "dev", ent2id, labels, args)
        test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)

    elif dataset == "chip":
        args.data_dir = './data/CHIP2020'
        args.max_seq_len = 150

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        if use_aug:
            train_data = get_data(processor, mid_data_path, "train_aug.json", "train", ent2id, labels, args)
        else:
            train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path,"chip_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "dev.json", "dev", ent2id, labels, args)
        # test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)

    elif dataset == "clue":
        args.data_dir = './data/CLUE'
        args.max_seq_len = 150

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        if use_aug:
            train_data = get_data(processor, mid_data_path, "train_aug.json", "train", ent2id, labels, args)
        else:
            train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path,"clue_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "dev.json", "dev", ent2id, labels, args)
        # test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)
    elif dataset == "addr":
        args.data_dir = './data/addr'
        args.max_seq_len = 64

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        if use_aug:
            train_data = get_data(processor, mid_data_path, "train_aug.json", "train", ent2id, labels, args)
        else:
            train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path,"clue_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "dev.json", "dev", ent2id, labels, args)
        # test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)
    elif dataset == "attr":
        args.data_dir = './data/attr'
        args.max_seq_len = 128

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        if use_aug:
            train_data = get_data(processor, mid_data_path, "train_aug.json", "train", ent2id, labels, args)
        else:
            train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path,"clue_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "dev.json", "dev", ent2id, labels, args)
        # test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)

    elif dataset == "sighan2005":
        args.data_dir = './data/sighan2005'
        args.max_seq_len = 512

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path, "clue_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "test.json", "dev", ent2id, labels, args)
        # test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)

    elif dataset == "gdcq":
        args.data_dir = './data/gdcq'
        args.max_seq_len = 70

        labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
        with open(labels_path, 'r') as fp:
            labels = json.load(fp)

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)
        id2ent = {v: k for k, v in ent2id.items()}

        mid_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        train_data = get_data(processor, mid_data_path, "train.json", "train", ent2id, labels, args)
        save_file(os.path.join(mid_data_path, "clue_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
        dev_data = get_data(processor, mid_data_path, "dev.json", "dev", ent2id, labels, args)
        # test_data = get_data(processor, mid_data_path, "test.json", "test", ent2id, labels, args)
