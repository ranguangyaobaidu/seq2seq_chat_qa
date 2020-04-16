from torch import nn
import torch
import torch.nn.functional as F


class CHATAttention(nn.Module):
    def __init__(self,method,hidden_size,batch_size):
        super(CHATAttention,self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(0.3)

        assert self.method in ['dot','general','concat'],'方法必须在dot,general,concat 中'

        if self.method == 'dot':
            pass
        elif self.method == 'general':
            self.lr = nn.Linear(hidden_size,hidden_size,bias=False)
        elif self.method == 'concat':
            self.Wa = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.Va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))

    def forward(self,encoder_output,decoder_hidden):
        """
        进行模型的构建
        encoder_output [batch_size,max_len,hidden_size]
        decoder_hidden [layers,batch_size,hidden_size]
        """
        batch_size, max_len,hidden_size = encoder_output.size()
        decoder_hidden = decoder_hidden[-1,:,:] # [batch_size,hidden_size]


        if self.method == "dot":
            return self.dot_score(decoder_hidden, encoder_output)
        elif self.method == "general":
            return self.general_score(decoder_hidden, encoder_output)
        elif self.method == "concat":
            return self.concat_score(decoder_hidden, encoder_output)

    def _dot_score(self,batch_size,max_len,decoder_hidden,encoder_output):
        attn_energies = torch.zeros(batch_size, max_len)
        for b in batch_size:
            for i in max_len:
                attn_energies[b, i] = decoder_hidden[b, :].dot(encoder_output[b, i])

        return F.softmax(attn_energies).unsqueeze(1)  # [batch_size,1,seq_len]



    def dot_score(self,decoder_hidden,encoder_output):
        """
        decoder_hidden  # [batch_size,hidden_size]
        encoder_output # [batch_size,max_len,hidden_size] max_len =1 用在gru后面
        """
        decoder_hidden = decoder_hidden.unsqueeze(-1) # [batch_size,hidden_size,1]
        attn_energies = torch.bmm(encoder_output, decoder_hidden)
        attn_energies = attn_energies.squeeze(-1)  # [batch_size,seq_len,1] ==>[batch_size,seq_len]

        return F.softmax(attn_energies,dim=-1).unsqueeze(1)  # [batch_size,1,seq_len]

    def general_score(self,decoder_hidden, encoder_output):
        """
        decoder_hidden  # [batch_size,hidden_size]
        encoder_output # [batch_size,max_len,hidden_size] max_len =1 用在gru后面
        """
        decoder_hidden = self.lr(decoder_hidden)
        decoder_hidden = decoder_hidden.unsqueeze(-1) # [batch_size,hidden_size,1]

        attn_energies = torch.bmm(encoder_output, decoder_hidden)
        attn_energies = attn_energies.squeeze(-1)  # [batch_size,seq_len,1] ==>[batch_size,seq_len]

        return F.softmax(attn_energies,dim=-1).unsqueeze(1)  # [batch_size,1,seq_len]

    def concat_score(self, decoder_hidden, encoder_output):
        """
        concat attention
        :param batch_size:int
        :param hidden: [batch_size,hidden_size]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        # 需要先进行repeat操作，变成和encoder_outputs相同的形状,让每个batch有seq_len个hidden_size
        x = decoder_hidden.repeat(encoder_output.size(1), 1, 1).permute(1,0,2)  ##[batch_size,seq_len,hidden_size]
        x = torch.tanh(self.Wa(torch.cat([x, encoder_output],dim=-1)))  # [batch_size,seq_len,hidden_size*2] --> [batch_size,seq_len,hidden_size]
        # va [batch_size,hidden_size] ---> [batch_size,hidden_size,1]
        attn_energis = torch.bmm(x, self.Va.unsqueeze(2))  # [batch_size,seq_len,1]
        attn_energis = attn_energis.squeeze(-1)
        # print("concat attention:",attn_energis.size(),encoder_outputs.size())
        return F.softmax(attn_energis, dim=-1).unsqueeze(1)  # [batch_size,1,seq_len]



