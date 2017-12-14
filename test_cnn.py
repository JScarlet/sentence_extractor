import json

import tensorflow as tf
from word_vec_generator import WordVecGenerator

path = 'test_data.json'
label_dict = {0: 'other', 1: 'Functionality and Behavior', 2: 'Concepts', 3: 'Directives'}

with open(path) as file_object:
    data_list = json.load(file_object)

word_generator = WordVecGenerator()
data_vec_list = word_generator.predict_data_list_extraction(data_list)
result = {}

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x')
    feed_dict = {x: data_vec_list}

    logits = graph.get_tensor_by_name('y_conv')
    classification_result = sess.run(logits, feed_dict)
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        data_list[i].setdefault('knowledge_pattern', label_dict.get(output[i]))
        print(data_list[i].get('text') + ' ' + label_dict.get([output[i]]))