import json
import sys

from sentence_parser import sentence2words, calculate_word_count, bm25

def calculate_average_length(sentences):
    count = 0
    for each_sentence in sentences:
        sentence_words = sentence2words(each_sentence)
        count += len(sentence_words)
    return count / len(sentences)

params = sys.argv
filename = "" + params[1]
with open(filename) as file_object:
    all_the_text = file_object.read()
    sentences_json, annotated_type_list_json = all_the_text.split("\t")

#print(sentences_json + "----------------------------------")
#print(annotated_type_list_json + "++++++++++++++++++++++++++++++++++")

sentences = json.loads(sentences_json)
#print(sentences)
annotated_type_list = json.loads(annotated_type_list_json)
#print(annotated_type_list)

average_length = calculate_average_length(sentences)

result_map = {}
for each_sentence in sentences:
    score_map = {}
    sentence_words = sentence2words(each_sentence)
    #print(sentence_words)
    for each in annotated_type_list:
        score = 0.0
        key = each.get("key")
        doc_count = each.get("doc_count")
        description_list = each.get("description_list")
        for each_word in sentence_words:
            word_count = calculate_word_count(each_word, description_list)
            #print(each_sentence, " ", average_length, " ", doc_count, " ", word_count)
            score += bm25(each_sentence, average_length, doc_count, word_count)
        score_map.setdefault(key, score)
    api_type = sorted(score_map, key=lambda x: score_map[x])[-1]
    result_map.setdefault(each_sentence, api_type)
print(result_map)

