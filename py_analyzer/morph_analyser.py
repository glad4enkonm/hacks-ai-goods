import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize
from stop_words import get_stop_words


class Analyser:
    def __init__(self):
        self.stop_words = set(get_stop_words('russian'))
        self.morph = pymorphy3.MorphAnalyzer()

    def to_keywords(self, sentence):
        sentence = sentence.replace(',', ' ').strip()
        for i in sent_tokenize(sentence):
            words_list = nltk.word_tokenize(i)
        return ' '.join(self.morph.parse(w.lower())[0].normal_form for w in words_list if
                        (not w in self.stop_words and w.isalpha()))
