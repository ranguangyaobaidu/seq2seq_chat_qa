import os
import sys
sys.path.insert(0,os.getcwd())

from num_model import NUMModel
import torch
from seq2seq_datsets import data_loader
from torch.optim import Adam
from torch.nn import NLLLoss
from num_sentence import NUMSentence
import config
import numpy as np



model = NUMModel()
optimizer = Adam(model.parameters())

criterion= NLLLoss(ignore_index=NUMSentence().PAD,reduction="mean")

def get_loss(decoder_outputs,target):
    #很多时候如果tensor进行了转置等操作，直接调用view进行形状的修改是无法成功的
    #target = target.contiguous().view(-1) #[batch_size*max_len]
    target = target.view(-1)
    decoder_outputs = decoder_outputs.view(config.BATCH_SIZE*config.MAX_LEN,-1)
    return criterion(decoder_outputs,target)


def train(epoch):
    for idx,(input,target,input_length,target_len) in enumerate(data_loader):
        optimizer.zero_grad()
        decoder_outputs,decoder_hidden = model(input,target,input_length)
        loss = get_loss(decoder_outputs,target)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(input), len(data_loader.dataset),100. * idx / len(data_loader), loss.item()))
        if idx % 100 == 0 and epoch >= 5:
            torch.save(model.state_dict(), "./model/seq2seq_num_model.pkl")
            torch.save(optimizer.state_dict(), './model/seq2seq_num_optimizer.pkl')

def evalute():
    model.eval()
    model.load_state_dict(torch.load("./model/seq2seq_num_model.pkl"))
    data = [str(i) for i in np.random.randint(0, 100000000, [10])]
    data = sorted(data, key=lambda x: len(x), reverse=True)
    print(data)

    data_len = torch.LongTensor([len(i) for i in data])
    model_input = torch.LongTensor([NUMSentence().transform(i,max_len=config.MAX_LEN) for i in data])

    result = model.evalute(model_input,data_len)
    value,index = result.topk(k=1)

    index = index.squeeze()
    array = index.detach().numpy()
    i = 0
    for line in array:
        sen = NUMSentence().reverse_transform(list(line))
        print('当前句子为：{}，预测句子为: {}'.format(data[i],sen))
        i += 1




if __name__ == '__main__':
    evalute()
