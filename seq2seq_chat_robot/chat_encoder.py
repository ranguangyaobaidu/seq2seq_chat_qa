import torch
from torch import nn
import chat_config



class CHATEncoder(nn.Module):
    def __init__(self):
        super(CHATEncoder,self).__init__()
        self.vocab_size = len(chat_config.ws)
        self.embedding_dim = chat_config.ENCODER_DIM

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=chat_config.ws.PAD)
        self.gru = nn.GRU(self.embedding_dim,hidden_size=chat_config.HIDDEN_SIZE,num_layers=chat_config.NUM_LAYERS,batch_first=True,dropout=chat_config.DROPOUT)

    def forward(self,data,data_len):
        """
        对输入进行encoder数据整理操作
        """
        embedd = self.embedding(data)
        # 对数据进行打包操作，加快计算速度
        embedd = nn.utils.rnn.pack_padded_sequence(embedd,lengths=data_len,batch_first=True)

        out,hidden = self.gru(embedd)
        # 对数据进行解包操作
        out,out_lengths = nn.utils.rnn.pad_packed_sequence(out,batch_first=True,padding_value=chat_config.ws.PAD)

        return out,hidden

