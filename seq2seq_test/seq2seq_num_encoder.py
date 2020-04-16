from torch import nn
import torch
from num_sentence import NUMSentence


class NUMEncoder(nn.Module):
    def __init__(self):
        super(NUMEncoder,self).__init__()
        self.vocab_size = len(NUMSentence())
        self.embedding_dim = 300
        self.embedding = nn.Embedding(self.vocab_size,embedding_dim=self.embedding_dim,padding_idx=NUMSentence().PAD)
        self.gru = nn.GRU(input_size=self.embedding_dim,hidden_size=32,num_layers=1,batch_first=True)

    def forward(self,data,data_len=None):
        """
        param: data 输入数据
        param: data_len 输入数据的真实长度
        """
        embeded = self.embedding(data)
        # 使用内置函数加快gru的运行速度
        # 对文本对齐之后的句子进行打包，能够加速在LSTM or GRU中的计算过程
        embeded = nn.utils.rnn.pack_padded_sequence(embeded,lengths=data_len,batch_first=True)

        # 使用gru进行计算
        out,hidden = self.gru(embeded)

        # 对打包后的结果进行解包
        out,out_length = nn.utils.rnn.pad_packed_sequence(out,batch_first=True,padding_value=NUMSentence().PAD)

        return out,hidden

if __name__ == '__main__':
    num = NUMEncoder()
    from seq2seq_datsets import data_loader
    for data,target,data_len,_ in data_loader:
        out,hidden = num(data,data_len)
        break
