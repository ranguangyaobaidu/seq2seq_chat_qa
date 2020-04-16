from torch import nn
import config
from num_sentence import NUMSentence
import torch.nn.functional as F
import torch
import random


class NUMDecoder(nn.Module):
    def __init__(self):
        super(NUMDecoder,self).__init__()
        self.max_seq_len = config.MAX_LEN
        self.vocab_size = len(NUMSentence())
        self.embedding_dim = 300

        self.embedding = nn.Embedding(self.vocab_size,embedding_dim=self.embedding_dim,padding_idx=NUMSentence().PAD)
        self.gru = nn.GRU(self.embedding_dim,hidden_size=32,num_layers=1,batch_first=True)
        self.logsoftmax = nn.LogSoftmax()
        self.lr = nn.Linear(32,self.vocab_size)

    def forward(self,encoder_hidden,target):
        # encoder_hidden [1,batch_size,hidden_size]
        # target [batch_size,max_len]

        # 初始化为sos,编码器的结果作为初始的隐层状态，定义一个[batch_size,1]的全为SOS的数据作为最开始的输入，告诉解码器，要开始工作了
        decoder_input = torch.LongTensor([[NUMSentence().SOS]]  * config.BATCH_SIZE)

        # 定义解码器的输出
        decoder_output = torch.zeros([config.BATCH_SIZE,config.MAX_LEN,self.vocab_size])

        # 定义decoder的隐藏层状态
        decoder_hidden = encoder_hidden

        # 进行每个word的计算
        for t in range(config.MAX_LEN):
            decoder_out,decoder_hidden = self.forward_step(decoder_input,decoder_hidden)

            # 将每个时间步上的数据进行拼接
            decoder_output[:,t,:] = decoder_out

            # 在训练过程中使用teacher forcing
            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                # 下一次的输入使用真实值
                decoder_input = target[:, t].unsqueeze(1)  # [batch_size,1]
            else:
                # 使用预测值，topk中k=1，即获取最后一个维度的最大的一个值
                value, index = torch.topk(decoder_out, 1)  # index [batch_size,1]
                decoder_input = index
        return decoder_output,decoder_hidden

    def evalute(self,encoder_hidden):
        """
        对数据进行预测
        """
        batch_size = encoder_hidden.size()[1]

        # 构造刚开始的输入
        decoder_input = torch.LongTensor([[NUMSentence().SOS]] * batch_size)
        decoder_hidden = encoder_hidden
        # 构造输入
        decoder_output = torch.zeros([batch_size,config.MAX_LEN,self.vocab_size])

        for t in range(config.MAX_LEN):
            out,decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
            decoder_output[:,t,:] = out
            value,index = out.topk(k=1) # [20,1]
            decoder_input = index
        return decoder_output


    def forward_step(self, decoder_input, decoder_hidden):
        """
        进行每个时间步的计算
        :param decoder_input:[batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :return: out:[batch_size,vocab_size],decoder_hidden:[1,batch_size,didden_size]
        """
        embeded = self.embedding(decoder_input) #embeded: [batch_size,1 , embedding_dim]

        out,hidden = self.gru(embeded,decoder_hidden) # out:[batch_size,1,hidden_size]

        # 去除Out在一维度的1
        out = out.squeeze(1) # out:[batch_size,hidden_size]

        out = F.log_softmax(self.lr(out),dim=-1) #out [batch_Size,1, vocab_size]

        return out,hidden



