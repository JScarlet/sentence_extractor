import nltk
from gensim.models import Word2Vec
from nltk import WordPunctTokenizer


class WordVecGenerator:
    stackoverflow_word2vec_model = None
    path = 'E:\Research\word2vec\model\so_word_embedding_model'

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
