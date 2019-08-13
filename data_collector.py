import csv
import json


def get_writing_exam(id):
    f = open("data/writing/writing.csv", errors='ignore')
    data_file = list(csv.reader(f, delimiter=','))
    f.close()
    topic = data_file[id][1]
    answer = []
    for row in data_file:
        if topic == row[1]:
            answer.append(dict(sample=data_file[id][2], band=data_file[id][3]))
    writing_exam = dict(
        topic=topic,
        answer=answer
    )
    return json.dumps(writing_exam)


# def map_data():
#     f = open("data/writing/writing.csv", errors='ignore')
#     data_file = list(csv.reader(f, delimiter=','))
#     f.close()
#     list_dict = []
#     for row in data_file:
#         for data in list_dict:
#             if row[1] == data['topic']:
#                 data['count'] += 1
#         list_dict.append(dict(topic=row[1], count=1))
#     for dic in list_dict:
#         if dic['count'] > 1: print(dic['topic'], dic['count'])
#     return list_dict
#
#
# map_data()
