import pickle
import chat_config



class ChatWordSeq2seq(object):
    """
    将chat数据集转换成数字
    """
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

        self.fited = False
        self.word_count = {}
        self.reverse_dict = {}

    def __len__(self):
        return len(self.dict)

    def fit(self,sentence):
        """
        对传入的句子进行处理
        """
        if not isinstance(sentence,str):
            raise Exception('输入str type')
        sentence = sentence.strip().split()

        for word in sentence:
            word =str(word)
            if word not in self.word_count:
                self.word_count[word] = 1
            else:
                self.word_count[word] += 1

        self.fited = True

    def bulit_vocab(self,min_count=1,max_count=None,max_features=None):
        assert self.fited is True,'请先fit()'
        if min_count is not None:
            self.word_count = {key:value for key,value in self.word_count.items() if value >= min_count}

        if max_count is not None:
            self.word_count = {key: value for key, value in self.word_count.items() if value <= max_count}

        if max_features is not None and isinstance(max_features,int):
            word_count_list = sorted(self.word_count.items(),key=lambda x:x[1],reverse=False)
            word_count_list = word_count_list[:max_features]
            for key,value in word_count_list:
                self.dict[key] = len(self.dict)
        else:
            for key in sorted(self.word_count.keys()):
                self.dict[key] = len(self.dict)

        # 构造index 到词的字典
        self.reverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None,add_eos=False):
        """
        将句子转换为数字
        """
        # 获取句子的长度
        if isinstance(sentence,list):
            seq_len = len(sentence) if add_eos is False else len(sentence) + 1
            # 将句子转换为数字
            result = [self.dict.get(i,self.UNK) for i in sentence]

            if max_len is not None:
                if max_len < seq_len:
                    if add_eos is True:
                        result = result[:max_len - 1]
                    else:
                        result = result[:max_len]
                else:
                    result += [self.PAD] * (max_len - seq_len)
            if add_eos is True:
                result += [self.EOS]

            return result

    def reverse_transform(self,array):
        """
        将数组转换为字符串
        """
        result = []
        array = list(array)

        for num in array:
            if num == self.EOS:
                break
            key = self.reverse_dict.get(num,self.UNK_TAG)
            result.append(key)

        # return [i for i in result if i not in [self.UNK_TAG,self.PAD_TAG]]
        return result






if __name__ == '__main__':
    word = ChatWordSeq2seq()

    word.fit('我 是 谁 你 是 谁')
    word.bulit_vocab()
    print(word.dict)
    print(word.reverse_dict)
    a1 = word.transform(['我','是','我','是'], max_len=4, add_eos=True)
    a2 = word.transform(['我','是','我','是'], max_len=4, add_eos=False)
    a3 = word.transform(['我','是','我','是','我','是'], max_len=4, add_eos=False)
    print(a1)
    print(a2)
    print(a3)
    print(word.reverse_transform(a1))
    print(word.reverse_transform(a2))
    print(word.reverse_transform(a3))

