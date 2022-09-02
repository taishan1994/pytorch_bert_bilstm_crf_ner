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


def cut_sentences_v3(sent):
    """以逗号进行分句"""
    sent = re.sub('([,，])([^”’])', r'\1\n\2', sent)
    return sent.split("\n")


def cut_sentences_main(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []
    if len(text) <= max_seq_len:
        return [text]

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    # print("sentences_v1=", sentences_v1)
    for sent_v1 in sentences_v1:
        # print(sent_v1)
        if len(sent_v1) > max_seq_len:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)
    # if ''.join(sentences) != text:
        # print(len(''.join(sentences)), len(text))

    res = []
    for sent in sentences:
        # print(sentences)
        if len(sent) > max_seq_len:
            sent_v3 = cut_sentences_v3(sent)
            # print(sent_v3)
            tmp = []
            length = 0
            for i in range(len(sent_v3)):
                if length + len(sent_v3[i]) < max_seq_len:
                    tmp.append(sent_v3[i])
                    length = length + len(sent_v3[i])
                else:
                    if "".join(tmp) != "":
                        res.append("".join(tmp))
                        tmp = [sent_v3[i]]
                        length = len(sent_v3[i])
            if "".join(tmp) != "":
                res.append("".join(tmp))
        else:
            res.append(sent)
    # assert ''.join(sentences) == text
    # 过滤掉空字符
    final_res = []
    for i in res:
        if i.strip() != "":
            final_res.append(i)
    return final_res