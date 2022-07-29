import re


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sent_for_bert(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    print("sentences_v1=", sentences_v1)
    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)

    assert ''.join(sentences) == text

    # 合并
    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]

        end_index_ = start_index_ + 1
        # 针对于bert模型，注意这里最大长度要减去2
        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_

        merged_sentences.append(tmp_text)

    return merged_sentences


def refactor_labels(sent, labels, start_index):
    """
    分句后需要重构 labels 的 offset
    :param sent: 切分并重新合并后的句子
    :param labels: 原始文档级的 labels
    :param start_index: 该句子在文档中的起始 offset
    :return (type, entity, offset)
    """
    new_labels = []
    end_index = start_index + len(sent)
    # _label： TI, 实体类别， 实体起始位置， 实体结束位置， 实体名）
    for _label in labels:
        if start_index <= _label[2] <= _label[3] <= end_index:
            new_offset = _label[2] - start_index
            if sent[new_offset: new_offset + len(_label[-1])] != _label[-1]:
                continue
            # assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

            new_labels.append((_label[1], _label[-1], new_offset))
        # label 被截断的情况
        elif _label[2] < end_index < _label[3]:
            raise RuntimeError(f'{sent}, {_label}')

    return new_labels


if __name__ == '__main__':
    raw_examples = [{
        "text": "深圳市沙头角保税区今后五年将充分发挥保税区的区位优势和政策优势，以高新技术产业为先导，积极调整产品结构，实施以转口贸易和仓储业为辅助的经营战略。把沙头角保税区建成按国际惯例运作、国内领先的特殊综合经济区域，使其成为该市外向型经济的快速增长点。",
        "labels": [
            [
                "T0",
                "GPE",
                0,
                3,
                "深圳市"
            ],
            [
                "T1",
                "GPE",
                3,
                6,
                "沙头角"
            ],
            [
                "T2",
                "LOC",
                6,
                9,
                "保税区"
            ],
            [
                "T3",
                "LOC",
                18,
                21,
                "保税区"
            ],
            [
                "T4",
                "GPE",
                73,
                76,
                "沙头角"
            ],
            [
                "T5",
                "LOC",
                76,
                79,
                "保税区"
            ]
        ]
    }]
    for i, item in enumerate(raw_examples):
        text = item['text']
        print(text[:90])
        sentences = cut_sent_for_bert(text, 90)
        start_index = 0

        for sent in sentences:
            labels = refactor_labels(sent, item['labels'], start_index)
            start_index += len(sent)

            print(sent)
            print(labels)
