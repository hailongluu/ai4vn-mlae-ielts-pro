import json


def get_score_reports(text):
    f = open('output.json')
    score_report = json.load(f)
    return score_report


def get_all_exam(id):
    f = open('full_exam.json')
    return json.load(f)
