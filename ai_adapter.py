import json
import re

from modules.coherence.coherence_model import CoherenceModel
from modules.vocab.model import Model
import data_collector
from modules.grammar.score_grammar import Essay


class APIAdapter:
    def __init__(self):
        self.coherence_model = CoherenceModel(weight_path="modules/coherence/pre-35-0.0187.h5",
                                              word_path="modules/coherence/word_index.pkl",
                                              matrix_path="modules/coherence/embedding_matrix.txt")
        self.vocab_model = Model()

    def get_score_reports(self, topic, text):
        coherence_score = self.coherence_model.predict(text, topic)
        vocab_score = self.vocab_model.predict(text)
        text = re.sub(r"\n", "", text)
        # print(text)

        grammar_report = Essay(text).toJSON()
        grammar_report = json.loads(grammar_report)
        grammar_score = grammar_report["score"]
        grammar_error = grammar_report["res_merged"]
        grammar_old_graph = grammar_report["org_paragraph"]
        grammar = dict(grammar_error=grammar_error, grammar_old_graph=grammar_old_graph)
        # grammar_error = [dict(length=0, offset=111, replacement="The")]
        vocab_recommend = [dict(length=0, offset=111, replacement="abc,abc,abc")]
        overall_score = (coherence_score + grammar_score + vocab_score) / 3
        overall_score = round(overall_score, 1)
        sample = data_collector.get_sample_topic(topic)
        score_report = dict(
            vocab_score=vocab_score,
            coherence_score=coherence_score,
            grammar_score=grammar_score,
            overall_score=overall_score,
            grammar=grammar,
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
