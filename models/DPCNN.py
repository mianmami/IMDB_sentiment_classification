import torch.nn as nn
import torch.nn.functional as F
from config import config


class Model(nn.Module):
    def __init__(self, embeddings):
        super(Model, self).__init__()
        if config.use_pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(config.max_vocab_dict_len, config.embedding_dim, padding_idx=0)
        self.conv_region = nn.Conv2d(1, config.dpcnn_num_filters, kernel_size=(3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.dpcnn_num_filters, config.dpcnn_num_filters, kernel_size=(3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # pad in top and bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # pad in bottom
        self.fc = nn.Linear(config.dpcnn_num_filters, config.num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(dim=1)  # [batch_size, 1, seq_len, embedding_dim]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = F.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = F.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]

        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px

        return x
