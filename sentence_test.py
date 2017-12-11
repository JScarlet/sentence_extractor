import json

dict ={'a':2,'b':3,'c':1}

print(sorted(dict,key=lambda x:dict[x])[-1])


file_object = open("C:\\Users\JScarlet\Desktop\\test.json", 'w')
result = []
for i in range(0, 3):
    temp = {}
    temp.setdefault("1", 1)
    temp.setdefault("2", 2)
    temp.setdefault("3", 3)
    result.append(temp)
json.dump(result, file_object)
file_object.close()


with open("C:\\Users\JScarlet\Desktop\\test.json") as load_f:
    s = json.load(load_f)
    print(type(s))
