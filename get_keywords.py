import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize
from stop_words import get_stop_words

stop_words = set(get_stop_words('russian'))
morph = pymorphy3.MorphAnalyzer()


def get_keywords(sentence):
    sentence = sentence.replace(',', ' ').strip()
    for i in sent_tokenize(sentence):
        words_list = nltk.word_tokenize(i)
    return ' '.join(morph.parse(w.lower())[0].normal_form for w in words_list if (not w in stop_words and w.isalpha()))
