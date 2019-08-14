import json
from modules.coherence.coherence_model import CoherenceModel


def get_score_reports(topic, text):
    coherence_model = CoherenceModel(weight_path="modules/coherence/weight.h5",
                                     word_path="modules/coherence/word_index.pkl",
                                     matrix_path="modules/coherence/embedding_matrix.txt")
    overall_score = coherence_model.predict(topic, text)
    print("overall_score: ", overall_score)
    f = open('output.json')
    score_report = json.load(f)
    return score_report


def get_all_exam(id):
    f = open('full_exam.json')
    return json.load(f)
