import json
import re

from modules.coherence.coherence_model import CoherenceModel
from modules.vocab.model import Model
import data_collector
from modules.grammar.score_grammar import Essay


class APIAdapter:
    def __init__(self):
        self.coherence_model = CoherenceModel(weight_path="modules/coherence/tuned-03-0.0161.h5",
                                              word_path="modules/coherence/word_index.pkl",
                                              matrix_path="modules/coherence/embedding_matrix.txt")
        self.vocab_model = Model()

    def get_score_reports(self, topic, text):
        topic = re.sub(r"\n", "", topic)

        coherence_score = self.coherence_model.predict(text, topic)
        vocab_score = self.vocab_model.predict(text)
        grammar_score = Essay(text).toJSON()
        # grammar_score = 0;
        grammar_error = [dict(length=0, offset=111, replacement="The")]
        vocab_recommend = [dict(length=0, offset=111, replacement="abc,abc,abc")]
        overall_score = (coherence_score + grammar_score + vocab_score) / 3
        overall_score = round(overall_score, 1)
        sample = data_collector.get_sample_topic(topic)
        score_report = dict(
            vocab_score=vocab_score,
            coherence_score=coherence_score,
            grammar_score=grammar_score,
            overall_score=overall_score,
            grammar_error=grammar_error,
            vocab_recommend=vocab_recommend,
            sample=sample
        )
        # print("overall_score: ", overall_score)
        # print("vocab_score:", vocab_score)
        # f = open('output.json')
        # score_report = json.load(f)
        return score_report

    def get_all_exam(self, id):
        f = open('full_exam.json')
        return json.load(f)
