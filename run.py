import torch
import pickle
import numpy as np
import argparse
from importlib import import_module
from config import config
from train_eval import train_eval


# --model 选择模型
# --pretrained 是否使用预训练词向量


parser = argparse.ArgumentParser(description='IMDB sentiment classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextRNN, TextCNN')
parser.add_argument('--pretrained', type=bool, default=False)
args = parser.parse_args()


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    embeddings = torch.tensor(np.load(config.embeddings_save_path + '.npz')['embeddings'].astype('float32'))

    # train_eval()
    config.use_pretrained_embedding = args.pretrained
    vocab_dict = pickle.load(open('./data/vocab_dict.pkl', 'rb'))
    config.max_vocab_dict_len = len(vocab_dict)  # 词表长度运行时赋值

    model = import_module('models.' + args.model).Model(embeddings)
    train_eval(model)
