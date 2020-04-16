from seq2seq_num_decoder import NUMDecoder
from seq2seq_num_encoder import NUMEncoder
from torch import nn

class NUMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NUMEncoder()
        self.decoder = NUMDecoder()

    def forward(self,data,target,data_len):
        encoder_out,encoder_hidden = self.encoder(data,data_len)
        decoder_out,decoder_hidden = self.decoder(encoder_hidden,target)

        return decoder_out,decoder_hidden

    def evalute(self,data,data_len):
        encoder_out,encoder_hidden = self.encoder(data,data_len)
        result = self.decoder.evalute(encoder_hidden)
        return result