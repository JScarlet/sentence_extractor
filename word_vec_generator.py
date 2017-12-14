import json

import numpy as np
from gensim.models import Word2Vec
from nltk import WordPunctTokenizer


class WordVecGenerator:
    stackoverflow_word2vec_model = None
    path = 'E:\Research\word2vec\model\so_word_embedding_model'
    label_dict = {'other': 0, 'Functionality and Behavior': 1, 'Concepts': 2, 'Directives': 3}

    def __init__(self):
        self.stackoverflow_word2vec_model = Word2Vec.load(self.path)
        print("done load word2vec")

    def sentence2words(self, sentence):
        result = []
        word_punct_tokenizer = WordPunctTokenizer()
        words = word_punct_tokenizer.tokenize(sentence)
        return words

    def generate_sentence_vector(self, sentence):
        sentence_vector = []
        words = self.sentence2words(sentence)
        for word in words:
            word_vector = self.get_word_vector(word)
            sentence_vector.append(word_vector)
        return sentence_vector

    def get_word_vector(self, word):
        try:
            word_vector = [value for value in self.stackoverflow_word2vec_model.wv[word]]
        except KeyError as ke:
            print(word + " doesn't exist in this vocabulary.")
            word_vector = [0 for i in range(0, 400)]
        return word_vector

    def data_list_extraction(self, path):
        labels = []
        descriptions = []
        with open(path) as file_object:
            data_list = json.load(file_object)
            for each in data_list:
                description_list = each.get('description_list')
                for each_description in description_list:
                    sentence_vector = self.generate_sentence_vector(each_description)
                    descriptions.append(sentence_vector)
                    labels.append(self.label_dict.get(each.get('key')))
        return np.asarray(descriptions, np.float32), np.asarray(labels, np.int32)

    def predict_data_list_extraction(self, data_list):
        descriptions = []
        for each in data_list:
            text = each.get('text')
            sentence_vector = self.generate_sentence_vector(text)
            descriptions.append(sentence_vector)
        return np.asarray(descriptions, np.float32)

