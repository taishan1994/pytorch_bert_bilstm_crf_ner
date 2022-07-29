# coding=utf-8
import random
import os
import json
import logging
import time
import pickle
import numpy as np
import torch



def timer(func):
    """
    函数计时器
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("{}共耗时约{:.4f}秒".format(func.__name__, end - start))
        return res

    return wrapper


def set_seed(seed=123):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    配置log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_json(data_dir, data, desc):
    """保存数据为json"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(data_dir, desc):
    """读取数据为json"""
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_pkl(data_dir, data, desc):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    """保存.pkl文件"""
    with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'wb') as f:
        pickle.dump(data, f)


def read_pkl(data_dir, desc):
    """读取.pkl文件"""
    with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'rb') as f:
        data = pickle.load(f)
    return data


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens
