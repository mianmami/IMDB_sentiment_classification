import torch.nn as nn
from config import config


class Model(nn.Module):
    def __init__(self, embeddings):
        super(Model, self).__init__()
        if config.use_pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=config.max_vocab_dict_len,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.text_rnn_hidden_size,
            num_layers=config.text_rnn_num_layer,
            batch_first=True,
            bidirectional=True,
            dropout=config.text_rnn_dropout
        )
        self.fc = nn.Linear(
            in_features=config.text_rnn_hidden_size * 2,
            out_features=config.num_classes
        )

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        out, _ = self.lstm(out)  # self.lstm(x) return: x, (h, c); x.shape is [batch_size, seq_len, hidden_size * 2]
        out = self.fc(out[:, -1, :])
        return out
