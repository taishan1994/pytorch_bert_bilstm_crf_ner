import os
import warnings
import json
import random


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def preprocess(input_path, save_path, mode, split=None, ratio=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result = []
    tmp = {}
    tmp['id'] = 0
    tmp['text'] = ''
    tmp['labels'] = []
    # =======先找出句子和句子中的所有实体和类型=======
    with open(input_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        texts = []
        words = []
        entities = []
        char_label_tmp = []
        for line in lines:
            line = line.strip().split(" ")
            if len(line) == 2:
                word = line[0]
                label = line[1]
                words.append(word)
                char_label_tmp.append(label)
            else:
                texts.append("".join(words))
                entities.append(get_entities(char_label_tmp))
                words = []
                char_label_tmp = []

    # ==========================================
    # =======找出句子中实体的位置=======
    # entities里面每一个元素：[实体类别, 实体起始位置, 实体结束位置]
    i = 0
    labels = set()
    for text, entity in zip(texts, entities):
        if entity:
            tmp['id'] = i
            tmp['text'] = text
            for j, ent in enumerate(entity):
                labels.add(ent[0])
                tmp['labels'].append(["T{}".format(str(j)), ent[0], ent[1], ent[2] + 1,
                                      text[int(ent[1]):int(ent[2] + 1)]])
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

    if mode == "train":
        label_path = os.path.join(save_path, "labels.json")
        with open(label_path, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(list(labels), ensure_ascii=False))


    if split:
        train_data_path = os.path.join(save_path, mode + ".json")
        dev_data_path = os.path.join(save_path, "dev" + ".json")
        random.shuffle(result)
        train_result = result[:int(len(result) * (1 - ratio))]
        dev_result = result[int(len(result) * (1 - ratio)):]
        with open(train_data_path, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(train_result, ensure_ascii=False))
        with open(dev_data_path, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(dev_result, ensure_ascii=False))
    else:
        data_path = os.path.join(save_path, mode + ".json")
        with open(data_path, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(result, ensure_ascii=False))


path = '../mid_data/'
preprocess("train.txt", path, "train", split=True, ratio=0.2)
# preprocess("train.txt", path, "train", split=None, ratio=None)
# preprocess("dev.txt", path, "dev", split=None, ratio=None)

labels_path = os.path.join(path, 'labels.json')
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
for k, v in enumerate(tmp_labels):
    label2id[v] = k

if not os.path.exists(path):
    os.makedirs(path)
with open(os.path.join(path, "nor_ent2id.json"), 'w') as fp:
    fp.write(json.dumps(label2id, ensure_ascii=False))
