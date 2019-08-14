import json
from modules.coherence.coherence_model import CoherenceModel


class APIAdapter:
    def __init__(self):
        self.coherence_model = CoherenceModel(weight_path="modules/coherence/tuned-03-0.0161.h5",
                                              word_path="modules/coherence/word_index.pkl",
                                              matrix_path="modules/coherence/embedding_matrix.txt")

    def get_score_reports(self, topic, text):
        overall_score = self.coherence_model.predict(text, topic)
        print("overall_score: ", overall_score)
        f = open('output.json')
        score_report = json.load(f)
        return score_report

    def get_all_exam(self, id):
        f = open('full_exam.json')
        return json.load(f)
