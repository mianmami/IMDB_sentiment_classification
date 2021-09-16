import re
import os
import torch
import pickle
import numpy as np
from config import config
from torch.utils.data import Dataset, DataLoader


PAD_TAG = '<PAD>'  # 填充字符
UNK_TAG = '<UNK>'  # 未知字符


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
            labels.append(0)

    # 读取正面情感文本
    pos_text_path = os.path.join(folder_path, 'pos')
    for file_name in os.listdir(pos_text_path):
        with open(os.path.join(pos_text_path, file_name), 'r', encoding='utf-8') as f:
            raw_text_list.append(f.read())
            labels.append(1)

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


def build_vocab_dict(sentences, stopwords_set, num_drop_high_freq=None, max_vocab_dict_len=None):
    """建立词表
    Args:
        sentences (list): [sentence, sentence, sentence, ...], 其中每个sentence是该句子分词后的单词列表
        num_drop_high_freq (int, optional): 词频表按词频从高到低排序, 去掉前多少个高频词. Defaults to None.
        max_vocab_dict_len (int, optional): 词表最大长度. Defaults to None.
        stopwords_set (set): 停用词集合
    Returns:
        vocab_dict (dict): 词表
    """
    word_freq = {}  # 初始化词频表
    vocab_dict = {  # 初始化词表
        PAD_TAG: 0,
        UNK_TAG: 1
    }

    # 构建词频表
    for sentence in sentences:
        for word in sentence:
            word_freq[word] = word_freq.get(word, 0) + 1
    word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

    # 过滤词频表
    if num_drop_high_freq is not None:
        word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[num_drop_high_freq:])
    if max_vocab_dict_len is not None:
        word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab_dict_len])

    # 根据词频表和停用词集合构建词表
    for word in word_freq:
        if word not in vocab_dict and word not in stopwords_set:
            vocab_dict[word] = len(vocab_dict)

    return vocab_dict


def transform(vocab_dict, sentence, max_seq_len):
    """将分词后的句子转成数字序列
    Args:
        vocab_dict (dict): 词表
        sentence (list): 分词后的句子组成的单词列表
        max_seq_len (int): 最大序列长度(用于统一句子长度)
    Returns:
        [list]: 转换后的数字序列
    """
    if len(sentence) < max_seq_len:
        sentence = sentence + [PAD_TAG] * (max_seq_len - len(sentence))
    else:
        sentence = sentence[:max_seq_len]

    return [vocab_dict.get(word, 1) for word in sentence]


def inverse_transform(vocab_dict, indices):
    """将数字序列转为单词序列
        Args:
            indices (list): 数字序列
        Returns:
            list: 单词序列
        """
    inverse_vocab_dict = {id: word for word, id in vocab_dict.items()}
    return [inverse_vocab_dict.get(id, UNK_TAG) for id in indices]


class ImdbDataset(Dataset):
    def __init__(self, train=True):
        with open(config.vocab_dict_save_path, 'rb') as f:
            self.vocab_dict = pickle.load(f)

        self.raw_texts, self.labels = read_raw_text(config.data_path, train)

    def __getitem__(self, index):
        text2ids = transform(self.vocab_dict, tokenizer(self.raw_texts[index]), max_seq_len=config.max_seq_len)
        label = self.labels[index]
        return text2ids, label

    def __len__(self):
        return len(self.raw_texts)


def collate_fn(batch):
    """对batch数据进行处理
    Args:
        batch (list): 执行batch_size次__getitem__()得到的结果, 在本例中其形式为 [(text2ids, label), (text2ids, label), ...]
    Returns:
        tensor
    """
    text2ids, labels = zip(*batch)
    return torch.LongTensor(text2ids), torch.LongTensor(labels)


def get_dataloader(train=True):
    """获取DataLoader
    Args:
        train (bool, optional): 是否加载训练集数据. Defaults to True.
    Returns:
        DataLoader
    """
    dataset = ImdbDataset(train)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, collate_fn=collate_fn)


if __name__ == '__main__':
    """创建词表和预训练向量表
    """
    if os.path.exists(config.vocab_dict_save_path):
        with open(config.vocab_dict_save_path, 'rb') as f:
            vocab_dict = pickle.load(f)
    else:
        stopwords_set = set()  # 构建停用词集合
        with open(config.stopwords_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                stopwords_set.add(line.strip())

        raw_text_list, _ = read_raw_text(data_path=config.data_path, train=True)
        sentences = []
        for raw_text in raw_text_list:
            sentences.append(tokenizer(raw_text))

        vocab_dict = build_vocab_dict(sentences, stopwords_set, num_drop_high_freq=config.num_drop_high_freq, max_vocab_dict_len=config.max_vocab_dict_len)

        with open(config.vocab_dict_save_path, 'wb') as f:
            pickle.dump(vocab_dict, f)

    embeddings = np.random.randn(len(vocab_dict), config.embedding_dim)  # 使用glove 300d的预训练词向量
    with open(config.pretrained_embeddings_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if line[0] in vocab_dict:
                embeddings[vocab_dict[line[0]]] = line[1:]
    np.savez_compressed(config.embeddings_save_path, embeddings=embeddings)
