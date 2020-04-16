from fasttext import FastText
#训练模型，设置n-garm=2
model = FastText.train_unsupervised(input="./datasets/word_a.txt",minCount=1,wordNgrams=2)
#获取句子向量，是对词向量的平均
model.save_model('./model/fasttext_test.pkl')