from torch import Tensor
import numpy as np
from collections import defaultdict

def get_entities(seq, text, suffix=False):
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
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            # chunks.append((prev_type, begin_offset, i-1))
            # 高勇：男，中国国籍，无境外居留权， 高勇：0-2，这里就为text[begin_offset:i]，如果是0-1，则是text[begin_offset:i+1]
            chunks.append((text[begin_offset:i+1],begin_offset,prev_type))
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

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

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

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def bioes_decode(decode_tokens, raw_text, id2ent):
    predict_entities = {}
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

            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, index_)]
            else:
                predict_entities[token_type].append((tmp_ent, int(index_)))

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

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent, int(start_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities
