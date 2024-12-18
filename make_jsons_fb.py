import json
import os



with open('C:/Users/user/Desktop/Project/datasets/fb/data/test.jsonl','r') as f:
    data = list(f)
    

list_data = []    

for point in data:
    point = json.loads(point)
    dict_temp = {}
    dict_temp['img'] = point['img'].split('/')[-1]
#     print (point['labels'])
#     if point['labels'][0]=='not harmful':
#         print("going")
#         dict_temp['label'] = 0
#     else:
#         dict_temp['label']  = 1
    dict_temp['label'] = point['label']
    dict_temp['text'] = point['text']
    list_data.append(dict_temp)
    

    
json_object = json.dumps(list_data, indent = 4)
  
# Writing to sample.json
with open("C:/Users/user/Desktop/Project/datasets/fb/data/files/test.json", "w") as outfile:
    outfile.write(json_object)