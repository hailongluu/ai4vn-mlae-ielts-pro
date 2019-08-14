from nltk import tokenize
from grammarbot import GrammarBotClient
from modules.grammar.txt2txt import DeepCorrect
import difflib
import pandas as pd
import json

client = GrammarBotClient()
client = GrammarBotClient(api_key='AF5B9M2X')
model_deep_correct = DeepCorrect('modules/grammar/params', 'modules/grammar/checkpoint')
print("Loaded model")
complex_words = ["who", "whom", "which", "that", "whose", "moreover", "therefore", "however", "while", ",", "such as",
                 "for example"]


def my_get_opcodes(a, b):
    sentence = a
    a = a.split()
    b = b.split()
    list_err = []
    s = difflib.SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != 'equal':
            ori_word = ''
            for oid in a[i1:i2]:
                ori_word += oid
                ori_word += ' '
            ori_word = ori_word[:-1]
            rep_word = ''
            for rid in b[j1:j2]:
                rep_word += rid
                rep_word += ' '
            rep_word = rep_word[:-1]
            if (rep_word[-2:] != 'ss'):
                offset = sentence.find(ori_word)
                length = len(ori_word)
                error_instace = Error(offset, length, rep_word)
                list_err.append(error_instace)

    return list_err


def tokenize_paragraph(paragraph):
    rs_paragraph = ''
    sentences = tokenize.sent_tokenize(paragraph)
    for sentence in sentences:
        rs_paragraph += sentence
        rs_paragraph += ' '
    return rs_paragraph


class Error:
    def __init__(self, offset, length, corrected_word):
        self.offset = offset
        self.length = length
        self.corrected_word = corrected_word

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Essay:
    def __init__(self, paragraph):
        self.org_paragraph = tokenize_paragraph(paragraph)
        res_api = self.api_check(self.org_paragraph)
        res_model = self.model_check(self.org_paragraph)
        self.res_merged = self.merger_res_err(res_api, res_model)
        self.score = self.score_essay()

    def print_err(self):
        print("API:")
        for err in self.res_api:
            print(err)
        print("Model:")
        for err in self.res_model:
            print(err)
        print("Result:")
        for err in self.res_merged:
            print(err)

    def api_check(self, org_paragraph):
        res = client.check(self.org_paragraph)
        res_api = []
        for err in res.matches:
            if (len(err.replacements) == 1):
                offset = err.replacement_offset
                length = err.replacement_length
                corrected_word = err.replacements[0]
                error_instace = Error(offset, length, corrected_word)
                res_api.append(error_instace)
        return res_api

    def model_check(self, org_paragraph):
        sentences = tokenize.sent_tokenize(org_paragraph)
        model_corrected_paragraph = ''
        for sentence in sentences:
            model_corrected_sentence = model_deep_correct.correct(sentence)
            model_corrected_paragraph += model_corrected_sentence[0][0]['sequence']
            model_corrected_paragraph += ' '
        return my_get_opcodes(org_paragraph, model_corrected_paragraph)

    def merger_res_err(self, res_api, res_model):
        res_merge = []
        for err_api in res_api:
            res_merge.append(err_api)
        for err_model in res_model:
            isIn = False
            for err in res_merge:
                if err_model.offset == err.offset:
                    isIn = True
                    break
            if isIn == False:
                res_merge.append(err_model)
        return res_merge

    def check_contain_complex_words(self, sentence):
        for word in complex_words:
            if (word in sentence):
                return True
        return False

    def calculate_number_words(self, sentence):
        return len(sentence.split())

    def score_a_sentence(self, sentence):
        score = 9
        if not self.check_contain_complex_words(sentence.lower()):
            score -= 1
        noWords = self.calculate_number_words(sentence)
        if (noWords < 15):
            score -= 1
        if (noWords < 8):
            score -= 1
        return score

    def score_essay(self):
        total_score = 0
        sentences = tokenize.sent_tokenize(self.org_paragraph)
        for (idx, sentence) in enumerate(sentences):
            score = self.score_a_sentence(sentence)
            total_score += score

        total_score -= (len(self.res_merged) * 2)
        number_words = self.calculate_number_words(self.org_paragraph)
        if number_words > 285 or number_words < 215:
            total_score -= 1

        return total_score / len(sentences)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def read_data_sets(file_path):
    data = pd.read_csv(file_path, sep='|')
    return data

# if __name__ == '__main__':
#     dataset = read_data_sets(file_path='test.csv')
#     paragraphs = dataset['review']
#     score_cons = dataset['sentiment'] * 9
#     for (idx, paragraph) in enumerate(paragraphs):
#         score_con = score_cons[idx]
#         grammar_score = 0
#         essay = Essay(paragraph)
#         print(essay.toJSON())
#         # print(str(score_con)+' '+str(grammar_score))
