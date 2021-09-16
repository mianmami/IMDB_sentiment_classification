import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import config
from utils import get_dataloader


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_correct_num(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train_eval(model):
    model = model.to(device)

    train_dataloader = get_dataloader(train=True)
    val_dataloader = get_dataloader(train=False)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        for texts, labels in tqdm(train_dataloader, total=len(train_dataloader)):
            texts = texts.to(device)
            labels = labels.to(device)

            preds = model(texts)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * config.batch_size
            train_correct += get_correct_num(preds, labels)

        train_loss_list.append(train_loss / 25000)
        train_accuracy_list.append(train_correct / 25000)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0.0
            for texts, labels in tqdm(val_dataloader, total=len(val_dataloader)):
                texts = texts.to(device)
                labels = labels.to(device)
                preds = model(texts)
                loss = F.cross_entropy(preds, labels)
                val_loss += loss.item() * config.batch_size
                val_correct += get_correct_num(preds, labels)
            val_loss_list.append(val_loss / 25000)
            val_accuracy_list.append(val_correct / 25000)

        print('epoch:{}/{}, train_loss:{}, train_accuracy:{}, val_loss:{}, val_accuracy:{}'.format((epoch + 1), config.epochs, train_loss / 25000, train_correct / 25000, val_loss / 25000, val_correct / 25000))

    plt.figure()
    plt.plot(list(range(1, config.epochs + 1)), train_loss_list, label='train loss')
    plt.plot(list(range(1, config.epochs + 1)), train_accuracy_list, label='train accuracy')
    plt.plot(list(range(1, config.epochs + 1)), val_loss_list, label='val loss')
    plt.plot(list(range(1, config.epochs + 1)), val_accuracy_list, label='val accuracy')
    plt.legend(loc='best')
    plt.show()
