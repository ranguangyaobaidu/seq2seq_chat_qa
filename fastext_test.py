import config
from fasttext import FastText

ft_model = FastText.load_model(config.FASTTEXT_CLASSIFIER_MODEL)

a = ft_model.get_sentence_vector("我 是 谁")
print(a)
textlist = [
    "人工智能 和 人工智障 有 啥 区别", #QA
    "我 来 学 python 是不是 脑袋 有 问题 哦", #QA
    "什么 是 python", #QA
    "人工智能 和 python 有 什么 区别",  #QA
    "为什么 要 学 python", #QA
    "python 该 怎么 学",  #CHAT
    "python", #QA
    "jave", #CHAT
    "php", #QA
    "理想 很 骨感 ，现实 很 丰满",
    "今天 天气 真好 啊",
    "你 怎么 可以 这样 呢",
    "哎呀 ， 我 错 了",
]
ret = ft_model.predict("你 怎么 可以 这样 呢")
print(ret)

ret = ft_model.predict(textlist[:3])
print(ret)