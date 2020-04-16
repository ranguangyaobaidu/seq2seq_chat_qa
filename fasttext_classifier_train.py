import logging
import fasttext
import config


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# ft_model = fasttext.train_supervised(config.FASTTEXT_CLASSIFIER_DATASETS,wordNgrams=1,epoch=20)
# ft_model.save_model(config.FASTTEXT_CLASSIFIER_MODEL)

ft_model = fasttext.train_supervised(config.FASTTEXT_CLASSIFIER_WORD_DATASETS,wordNgrams=1,epoch=20)
ft_model.save_model(config.FASTTEXT_CLASSIFIER_WORD_MODEL)