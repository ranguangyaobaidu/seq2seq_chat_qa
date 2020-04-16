from chat_model import CHATModel
import torch.nn.functional as F
from torch.optim import Adam
import chat_config
from chat_datasets import data_loader
import torch
from torch import nn


model = CHATModel()
optimizer = Adam(model.parameters())

# model.load_state_dict(torch.load("model/chat_word_punctuation_model.pkl"))


def get_loss(decoder_out,target):
    """
    计算损失
    decoder_out [batch_size,max_len,vocab_size]
    target [batch_size,max_len]
    """
    target = target.view(-1)
    decoder_out = decoder_out.view([decoder_out.size(0) * chat_config.MAX_LEN,-1])

    return F.nll_loss(decoder_out,target)


def train(epochs):
    """
    进行训练
    """
    for epoch in range(epochs):

        for idx,(data,target,data_len) in enumerate(data_loader):
            optimizer.zero_grad()

            # 输入模型进行计算
            decoder_output = model(data,target,data_len)
            loss = get_loss(decoder_output,target)

            loss.backward()
            # 进行梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(),5)

            optimizer.step()

            if idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(data), len(data_loader.dataset),
                           100. * idx / len(data_loader), loss.item()))
            if epoch >= 6 and idx % 100 == 0 and idx != 0:
                torch.save(model.state_dict(), "model/chat_word_punctuation_model.pkl")
                torch.save(optimizer.state_dict(), "model/chat_word_punctuation_optimizer.pkl")




if __name__ == '__main__':
    train(10)

