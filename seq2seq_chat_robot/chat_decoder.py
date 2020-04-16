from torch import nn
import chat_config
import torch
import torch.nn.functional as F
import numpy as np
from chat_attention import CHATAttention



class CHATDecoder(nn.Module):
    def __init__(self):
        super(CHATDecoder,self).__init__()
        self.vocab_size = len(chat_config.ws)
        self.embedding_dim = chat_config.ENCODER_DIM

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=chat_config.ws.PAD)
        self.gru = nn.GRU(self.embedding_dim,hidden_size=chat_config.HIDDEN_SIZE,num_layers=chat_config.NUM_LAYERS,batch_first=True,dropout=chat_config.DROPOUT)
        self.lr = nn.Linear(chat_config.HIDDEN_SIZE,self.vocab_size)
        self.lr_concat = nn.Linear(chat_config.HIDDEN_SIZE * 2,chat_config.HIDDEN_SIZE)
        self.normal = nn.BatchNorm1d(chat_config.HIDDEN_SIZE)
        self.attention = CHATAttention(method=chat_config.METHOD, hidden_size=chat_config.HIDDEN_SIZE,batch_size=chat_config.BATCH_SIZE)

    def forward(self,encoder_hidden,target,encoder_output):
        """
        构建模型
        """
        # 获取数据batch_size
        batch_size = encoder_hidden.size()[1]

        # 构建decoder的输入输出，以及开始标志
        decoder_input = torch.LongTensor([[chat_config.ws.SOS]] * batch_size)
        decoder_output = torch.zeros([batch_size,chat_config.MAX_LEN,self.vocab_size])
        decoder_hidden = encoder_hidden

        # 对每个时间步上的数据进行处理
        for t in range(chat_config.MAX_LEN):
            decoder_output_t,decoder_hidden = self._forward_step_match(decoder_input,decoder_hidden,encoder_output)

            # 将输出的结果进行拼接
            decoder_output[:,t,:] = decoder_output_t

            # 进行优化下一次的输入，使用teacher forcing
            teacher_rate = np.random.random() > 0.5
            if teacher_rate:
                # 使用teacher
                decoder_input = target[:,t].unsqueeze(1)
            else:
                value,index = decoder_output_t.topk(k=1)
                decoder_input = index

        return decoder_output,decoder_hidden

    def predict(self,encoder_hidden,encoder_output):

        # 获取数据batch_size
        batch_size = encoder_hidden.size()[1]

        # 构建decoder的输入输出，以及开始标志
        decoder_input = torch.LongTensor([[chat_config.ws.SOS]] * batch_size)
        decoder_output = torch.zeros([batch_size, chat_config.MAX_LEN, self.vocab_size])
        decoder_hidden = encoder_hidden

        for t in range(chat_config.MAX_LEN):
            decoder_output_t, decoder_hidden = self._forward_step_match(decoder_input, decoder_hidden,encoder_output)

            # 将输出的结果进行拼接
            decoder_output[:, t, :] = decoder_output_t

            value, index = decoder_output_t.topk(k=1)
            decoder_input = index

        return decoder_output, decoder_hidden


    def _forward_step_match(self,decoder_input,decoder_hidden,encoder_output):
        """
        对每个词进行计算
        """
        # 对输入数据进行处理
        embeded = self.embedding(decoder_input)
        # out [batch_size,1,hidden_size]  decoder_hidden [2,batch_size,hidden_size]
        out,decoder_hidden = self.gru(embeded,decoder_hidden)
        # 进行形状变化
        out = out.squeeze(1) # [batch_size,hidden_size]

        # 使用attention机制
        attn_weights = self.attention(encoder_output,decoder_hidden) # attn_weights [batch_size,1,seq_len]

        # attn_weights [batch_size,1,seq_len] * [batch_size,seq_len,hidden_size]
        context = attn_weights.bmm(encoder_output)  # [batch_size,1,hidden_size]

        context = context.squeeze(1)  # [batch_size,hidden_size]

        # 将content与decoder的输出结果合并，作为下一次的输入
        context = torch.cat((out, context), 1)  # [batch_size,hidden_size*2]

        context = torch.tanh(self.lr_concat(context))  # [batch_size,hidden_size]

        concat_output = self.lr(context)

        # 计算sofamax
        return F.log_softmax(concat_output,dim=-1),decoder_hidden