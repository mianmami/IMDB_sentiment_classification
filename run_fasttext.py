import os
import re
import fasttext
import pandas as pd
from config import config


def read_raw_text(data_path, train=True):
    """读取原始文本数据
    Args:
        data_path (string): 数据路径
        train (bool, optional): 是否读取训练集, False表示读取测试集. Defaults to True.
    Returns:
        raw_text_list (list): 原始文本数据列表, 列表中的每一项就是一条评论
        labels (list): 标签列表, neg - 0, pos - 1
    """
    if train:
        folder_path = os.path.join(data_path, 'train')
    else:
        folder_path = os.path.join(data_path, 'test')

    raw_text_list = []
    labels = []

    # 读取负面情感文本
    neg_text_path = os.path.join(folder_path, 'neg')
    for file_name in os.listdir(neg_text_path):
        with open(os.path.join(neg_text_path, file_name), 'r', encoding='utf-8') as f:
            raw_text_list.append(f.read())
            labels.append('__label__negative')

    # 读取正面情感文本
    pos_text_path = os.path.join(folder_path, 'pos')
    for file_name in os.listdir(pos_text_path):
        with open(os.path.join(pos_text_path, file_name), 'r', encoding='utf-8') as f:
            raw_text_list.append(f.read())
            labels.append('__label__positive')

    return raw_text_list, labels


def tokenizer(raw_sentence):
    """分词器
    Args:
        raw_sentence (string): 原始句子
    Returns:
        [list]: 分词后的单词组成的列表
    """
    filters = [
        '!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.',
        '/', ':', ';', '<', '=', '>', '\?', '@', '\[', '\\', '\]', '^', '_',
        '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“'
    ]

    raw_sentence = raw_sentence.lower()
    raw_sentence = re.sub('<br />', ' ', raw_sentence)  # 去掉句子中的特殊字符
    raw_sentence = re.sub('|'.join(filters), ' ', raw_sentence)

    return [word for word in raw_sentence.strip().split(' ') if len(word) > 0]


def run_fasttext():
    # 将数据处理成fasttext要求的标准格式, fasttext要求的标准格式: 标签以__label__开头, 放在每一行的最前面, 所有字段都使用空格隔开.
    if not os.path.exists(config.train_csv_path):
        train_raw_texts, train_labels = read_raw_text(config.data_path, train=True)

        train_documents = []
        for idx in range(len(train_raw_texts)):
            train_documents.append([train_labels[idx] + ' ' + ' '.join(tokenizer(train_raw_texts[idx]))])

        pd.DataFrame(data=train_documents).to_csv(config.train_csv_path, encoding='utf-8', index=False, header=False)

    if not os.path.exists(config.test_csv_path):
        test_raw_texts, test_labels = read_raw_text(config.data_path, train=False)

        test_documents = []
        for idx in range(len(test_raw_texts)):
            test_documents.append([test_labels[idx] + ' ' + ' '.join(tokenizer(test_raw_texts[idx]))])

        pd.DataFrame(data=test_documents).to_csv(config.test_csv_path, encoding='utf-8', index=False, header=False)

    # 使用 facebook research 的 fastText 进行训练
    model = fasttext.train_supervised(input=config.train_csv_path)
    # 测试
    score = model.test(path=config.test_csv_path)
	# print(score)  # model.test() return (num_samples, precision, recall)

    print('fastText accuracy: {}'.format(score[1]))
