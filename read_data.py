import json

import sys

params = sys.argv
filename = params[1]
with open(filename) as file_object:
    data_list = json.load(file_object)
print(data_list)