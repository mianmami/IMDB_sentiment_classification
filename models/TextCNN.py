import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=config.text_cnn_num_filters, kernel_size=(k, config.embedding_dim)) for k in config.text_cnn_kernel_sizes
        ])
        self.dropout = nn.Dropout(config.text_cnn_dropout)
        self.fc = nn.Linear(in_features=len(config.text_cnn_kernel_sizes) * config.text_cnn_num_filters, out_features=config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(dim=3)  # [batch_size, num_filters, conv_seq_length]
        # max_pool1d's kernel size is 1 * kernel_size
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(dim=2)  # [batch_size, num_filters]
        return x

    def forward(self, out):
        out = self.embedding(out).unsqueeze(dim=1)  # [batch_size, 1, seq_length, embeddimg_dim]
        out = [self.conv_and_pool(out, conv) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [batch_size, len(kernel_size) * num_filters]
        out = self.dropout(out)
        out = self.fc(out)
        return out
