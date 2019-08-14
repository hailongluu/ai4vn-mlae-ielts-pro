import csv
import json
import re
import contractions


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = str(string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r" ", "", string)

    string = contractions.fix(string)
    # print(string)
    string = string.translate(str.maketrans('', '', "!#$%&'?().,*.+-/:<=>@[\]^_`{|}~"))
    return string.strip().lower()


def get_sample_topic(topic):
    topic = clean_str(topic)
    f = open("data/writing/writing.csv", errors='ignore')
    data_file = list(csv.reader(f, delimiter=','))
    f.close()
    sample = []
    for row in data_file:
        print(row[0])
        print(topic)
        topic_sample = clean_str(row[1])
        print(topic_sample)
        if topic == topic_sample:
            sample.append(dict(text=row[2], band=row[3]))
    return sample


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
