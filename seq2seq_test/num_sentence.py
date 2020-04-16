import string
import re


class NUMSentence(object):
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG:self.PAD,
            self.UNK_TAG:self.UNK,
            self.SOS_TAG:self.SOS,
            self.EOS_TAG:self.EOS
        }

        # 构造所有的次数
        for i in range(10):
            self.dict[str(i)] = len(self.dict)

        # 构造反序列化字典
        self.reverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def __len__(self):
        return len(self.dict)

    def transform(self,sentence,max_len=None,add_eos=False):
        """
        将句子转换为数字
        """
        sentence = sentence.strip()
        sentence = [self.dict.get(i,self.UNK) for i in sentence]
        # 获取句子的长度
        seq_len = len(sentence) if add_eos is False else len(sentence) + 1
        if max_len and add_eos is not None:
            assert max_len >= seq_len, "max_len 需要大于seq+eos的长度"
        if add_eos:
            if max_len is not None:
                sentence += [self.PAD] * (max_len - seq_len)

            sentence += [self.EOS]
        else:
            if max_len is not None:
                sentence += [self.PAD] * (max_len - seq_len)

        return sentence

    def reverse_transform(self,sentence):
        """
        将数字转换为字符串
        """
        result = []
        for word in sentence:
            if word == self.EOS:
                break

            value = self.reverse_dict.get(word,self.UNK)
            result.append(value)

        return ''.join([i for i in result if i not in [self.PAD_TAG]])








if __name__ == '__main__':
    num = NUMSentence()
    print(num.dict)
    print(num.reverse_dict)
    a = num.transform('4343122341',max_len=11,add_eos=True)
    print(len('4343122341'))
    print(len(a))
    print(a)
    print(num.reverse_transform(a))