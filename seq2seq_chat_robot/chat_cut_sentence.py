import os
import sys
sys.path.insert(0,os.getcwd())
import jieba
import string
import logging

jieba.setLogLevel(log_level=logging.INFO)


class CHATCUTSENtence(object):
    def __init__(self):
        self.letters = string.ascii_letters
        self.jieba_util = jieba
        self.filters= [",","-","."," ",'{','}','[',']','~','，']

    def _cut_by_word(self,sentence,punctuation):
        """
        进行单个词切分,同时不对英文处理
        """
        if isinstance(sentence,str):
            sentence = sentence.strip()
        if punctuation is True:
            # 保留标点符号
            return ' '.join([i for i in list(sentence.strip()) if i != ' '])

        temp = ''
        res = []
        for word in sentence:
            if word in self.filters:
                if temp != "":
                    res.append(temp)
                    temp = ''
                continue

            if word in self.letters:
                temp += word.lower()
            else:
                if temp != '':
                    res.append(temp)
                    temp = ''
                else:
                    res.append(word)
        if temp != "":
            if not res:
                res.append(temp)
                return ' '.join(res)
            else:
                res.append(temp)

        return " ".join(res)



    def cut_sentence(self,sentence,by_word=True,with_sg=False,punctuation=False):
        """
        对句子进行切分

        """
        if len(sentence) <= 0:
            return
        assert by_word is True or with_sg is False,'单个词分词没有词性'

        if by_word is True:
            return self._cut_by_word(sentence,punctuation)

        # 进行结巴分词
        content = self.jieba_util.lcut(sentence)
        return ' '.join(content)





if __name__ == '__main__':
    cut = CHATCUTSENtence()
    print(cut.cut_sentence('我很难过 ，安慰我 ~',by_word=True,punctuation=True))
    print(cut.cut_sentence('我很难过 ，安慰我 ~',by_word=True,punctuation=False))