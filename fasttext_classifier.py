import fasttext
from cut_sentence import CUTSentence
import config


class FastTextClassifier(object):
    """
    对数据进行目标识别 是问题question 还是闲聊chat
    """
    def __init__(self):
        self.cut_sentence = CUTSentence()
        self.ft_model_word = fasttext.load_model(config.FASTTEXT_CLASSIFIER_WORD_MODEL)
        self.ft_model = fasttext.load_model(config.FASTTEXT_CLASSIFIER_MODEL)
        self.text_class = {'__label__QA':0,'__label__chat':1}

    def predict(self,sentences,cut_centrain=True):
        """
        判断该句子是什么类型
        """
        if isinstance(sentences,list):
            if cut_centrain is False:
                lines = []
                lines_word = []
                for sentence in sentences:
                    # 处理通过jieba分词的数据
                    line = " ".join(self.cut_sentence.cut_sentence(sentence, by_word=False, with_sg=False))
                    lines.append(line)
                    line_word = " ".join(self.cut_sentence.cut_sentence(sentence, by_word=True, with_sg=False))
                    lines_word.append(line_word)

                result = self.ft_model.predict(lines)
                result_word = self.ft_model_word.predict(lines_word)
                res = self.handle_result_list(result, result_word)
                return res

            else:
                result = self.ft_model.predict(sentences)
                result_word = self.ft_model_word.predict(sentences)
                res = self.handle_result_list(result,result_word)
                return res

        elif isinstance(sentences,str):
            result,result_word = self._predict(sentences)
            acc,label,word_label,word_acc = self.handle_result_str(result,result_word)
            ret = self.judge_chat_question(acc,label,word_label,word_acc)
            return ret

        else:
            raise Exception('请输入列表或字符串')


    def _predict(self,sentence):
        """
        进行预测
        """
        line = " ".join(self.cut_sentence.cut_sentence(sentence, by_word=False, with_sg=False))
        result = self.ft_model.predict(line)

        # 处理通过单个词进行分词的数据
        line = " ".join(self.cut_sentence.cut_sentence(sentence, by_word=True, with_sg=False))
        result_word = self.ft_model_word.predict(line)

        return result,result_word



    def handle_result_list(self,result,result_word):
        """
        对预测的结果进行处理
        """
        label_list = result[0]
        acc_list = result[1]
        word_label_list = result_word[0]
        word_acc_list = result_word[1]

        i = 0
        res = []
        while i < len(label_list) and len(label_list) == len(acc_list):

            acc = acc_list[i]
            label = label_list[i][0]

            word_acc = word_acc_list[i]
            word_label = word_label_list[i][0]

            i += 1


            if word_label == label:

                res.append(self.text_class[label])
            else:
                res.append(self.text_class[label if acc > word_acc else word_label])

        return res

    def handle_result_str(self,result,result_word):
        """
        对预测的结果进行处理
        """
        label = result[0][0]
        acc = result[1][0]
        word_label = result_word[0][0]
        word_acc = result_word[1][0]

        return acc,label,word_label,word_acc


    def judge_chat_question(self,acc,label,word_label,word_acc):
        """
        对准确率进行判断
        """
        if acc > 0.95 or word_acc > 0.95:
            # 是QA
            if label == word_label:
                return 'label : {}, acc : {}'.format(label,acc if acc > word_acc else word_acc),
            else:
                return 'label : {}, acc : {}'.format(label if acc > word_acc else word_label,acc if acc > word_acc else word_acc)
        else:
            return '不能预测该句子'



if __name__ == '__main__':
    f = FastTextClassifier()
    textlist = [
        "人工智能 和 人工智障 有 啥 区别",  # QA
        "我 来 学 python 是不是 脑袋 有 问题 哦",  # QA
        "什么 是 python",  # QA
        "人工智能 和 python 有 什么 区别",  # QA
        "为什么 要 学 python",  # QA
        "python 该 怎么 学",  # CHAT
        "python",  # QA
        "jave",  # CHAT
        "php",  # QA
        "理想 很 骨感 ，现实 很 丰满",
        "今天 天气 真好 啊",
        "你 怎么 可以 这样 呢",
        "哎呀 ， 我 错 了",
    ]
    print(f.predict(['人工智能和人工智障有啥区别'],cut_centrain=False))
    print(f.predict(textlist))