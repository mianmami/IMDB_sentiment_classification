class Config:
    def __init__(self):
        self.data_path = './data/aclImdb'  # 数据路径
        self.vocab_dict_save_path = './data/vocab_dict.pkl'  # 词表保存路径
        self.pretrained_embeddings_path = './data/glove.6B/glove.6B.300d.txt'  # glove 300d预训练词向量路径
        self.embeddings_save_path = './data/embeddings'  # 使用预训练词向量的词表保存路径
        self.stopwords_path = './data/stopwords.en.txt'  # 停用词路径
        self.train_csv_path = './data/train.csv'  # fasttext 需要的语料库标准格式 train
        self.test_csv_path = './data/test.csv'  # fasttext 需要的语料库标准格式 test

        self.num_drop_high_freq = 20  # 剔除词频最高的前20个词
        self.max_vocab_dict_len = 10000  # 词表最大长度
        self.max_seq_len = 200  # 统一序列长度

        self.batch_size = 256  # batch size
        self.shuffle = True  # is shuffled
        self.embedding_dim = 300  # embedding dim
        self.num_classes = 2  # number of classes
        self.learning_rate = 1e-3
        self.epochs = 10
        self.use_pretrained_embedding = False  # 是否使用预训练词向量

        # TextRNN Config
        self.text_rnn_hidden_size = 128  # feature number of hidden state
        self.text_rnn_num_layer = 2  # layer num of lstm
        self.text_rnn_dropout = 0.5  # drop out rate

        # TextCNN Config
        self.text_cnn_kernel_sizes = [2, 3, 4]
        self.text_cnn_num_filters = 100
        self.text_cnn_dropout = 0.1

        # DPCNN Config
        self.dpcnn_num_filters = 250


config = Config()
