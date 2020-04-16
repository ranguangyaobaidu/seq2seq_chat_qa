from chat_encoder import CHATEncoder
from chat_decoder import CHATDecoder
from torch import nn


class CHATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CHATEncoder()
        self.decoder = CHATDecoder()

    def forward(self,data,target,data_len):
        """
        进行模型的构建
        """
        encoder_output,encoder_hidden = self.encoder(data,data_len)
        decoder_output,decoder_hidden = self.decoder(encoder_hidden,target,encoder_output)

        return decoder_output

    def predict(self,data,data_len):

        encoder_output, encoder_hidden = self.encoder(data, data_len)
        decoder_output, decoder_hidden = self.decoder.predict(encoder_hidden,encoder_output)

        return decoder_output