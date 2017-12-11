import math
import nltk
import json
from nltk import WordPunctTokenizer

from annotated_api_text import AnnotatedAPIText
from api_sentence import APISentence
from api_text import APIText
from data_source import DataSource
from document import Document


def write_data(annotated_api_text_list):
    result = []
    file_object = open('bm25.json', 'w')
    annotated_api_text_map = annotated_api_text_list_extraction(annotated_api_text_list)
    for key in annotated_api_text_map.keys():
        temp = {}
        description_list = annotated_api_text_map.get(key)
        doc_count = len(description_list)
        temp.setdefault("key", key)
        temp.setdefault("doc_count", doc_count)
        temp.setdefault("description_list", description_list)
        result.append(temp)
    json.dump(result, file_object)
    file_object.close()


def read_data(filename):
    with open(filename) as file_object:
        data_list = json.load(file_object)
    return data_list

def calculate(data_source, annotated_type_map_list):
    api_sentences = data_source_extraction(data_source.document_list)
    average_length = calculate_average_length(api_sentences)
    for each_api_sentence in api_sentences:
        sentence_text = each_api_sentence.api_sentence
        score_map = {}
        for each in annotated_type_map_list:
            score = 0.0
            key = each.get("key")
            doc_count = each.get("doc_count")
            description_list = each.get("description_list")
            sentence_words = sentence2words(sentence_text)
            for each_word in sentence_words:
                word_count = calculate_word_count(each_word, description_list)
                print(sentence_text, " ", average_length, " ", doc_count, " ", word_count)
                score += bm25(sentence_text, average_length, doc_count, word_count)
            score_map.setdefault(key, score)

        api_type = sorted(score_map,key=lambda x:score_map[x])[-1]
        each_api_sentence.type = api_type
    return api_sentences

def data_source_extraction(data_source):
    result = []
    for i in range(0, len(data_source)):
        for j in range(0, len(data_source[i].api_text_list)):
            api_name = data_source[i].api_text_list[j].api_name
            paragraph = data_source[i].api_text_list[j].api_description
            sentences = doc2sentences(paragraph)
            for k in range(0, len(sentences)):
                api_sentence = APISentence(api_name, sentences[k], i, j, k)
                result.append(api_sentence)
    return result


def annotated_api_text_list_extraction(annotated_api_text_list):
    result = {}
    for each in annotated_api_text_list:
        type = each.type
        api_text = each.api_text
        description = api_text.api_description
        if type in result:
            description_list = result.get(type)
            description_list.append(description)
            result.setdefault(type, description_list)
        else:
            result.setdefault(type, [])
    return result


def calculate_average_length(api_sentences):
    count = 0
    for each in api_sentences:
        sentence = each.api_sentence
        sentence_words = sentence2words(sentence)
        count += len(sentence_words)
        print(count, " ", len(api_sentences))
    return count / len(api_sentences)


def calculate_word_count(word, description_list):
    result = 0
    for sentence in description_list:
        if word_in_sentence(word, sentence):
            result += 1
    return result


def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]

def sentence2words(sentence):
    result = []
    word_punct_tokenizer = WordPunctTokenizer()
    words = word_punct_tokenizer.tokenize(sentence)
    stemmer = nltk.stem.SnowballStemmer('english')
    for word in words:
        ori_word = stemmer.stem(word)
        result.append(ori_word)
    return result


def doc2sentences(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def frequency(sentence):
    word_map = {}
    words = sentence2words(sentence)
    for word in words:
        if word in word_map.keys():
            word_number = word_map.get(word)
            word_number += 1
            word_map.setdefault(word, word_number)
        else:
            word_map.setdefault(word, 1)
    return word_map


def bm25(sentence, avgsl, doc_count, word_count, k1=1.5, b=0.75):
    word_map = frequency(sentence)
    score = 0.0
    for word in word_map.keys():
        word_frequency = word_map.get(word)
        score += calculate_idf(doc_count, word_count) * (word_frequency / len(word_map)) * (k1 + 1) / ((word_frequency / len(word_map) + k1 * (1 - b + b * len(sentence2words(sentence)) / avgsl)))
    return score


def calculate_idf(doc_count, word_count):
    return math.log((doc_count - word_count + 0.5) / (word_count + 0.5))


def word_in_sentence(word, sentence):
    if word in sentence:
        return True
    else:
        words = sentence2words(sentence)
        for each in words:
            similar = similarity(word, each)
            if similar > 0.7:
                return True
        return False

def similarity(source, target):
    dl_distance = damerau_levenshtein_distance(source, target)
    max_length = max(len(source), len(target))
    relative_distance = 1 - (dl_distance / max_length)
    return relative_distance


if __name__ == "__main__":
    api_text1 = APIText("api1", "description1")
    api_text2 = APIText("api2", "description2")
    api_text_list = [api_text1, api_text2]
    document = Document(api_text_list)
    document_list = [document]
    data_source = DataSource(document_list)
    annotated_api_text1 = AnnotatedAPIText(api_text1, 1)
    annotated_api_text2 = AnnotatedAPIText(api_text2, 2)
    annotated_api_text_list = [annotated_api_text1, annotated_api_text2]
    #result = calculate(data_source, annotated_api_text_list)
    #for each in result:
    #    print(each.type)
    data_list = read_data("C:\\Users\\JScarlet\Desktop\\test.json")
    print(data_list)
