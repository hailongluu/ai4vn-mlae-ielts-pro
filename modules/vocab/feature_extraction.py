import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import os
from io_util import load_data, dump_data


big_data = open('big.txt').read()
words_ = re.findall('[a-z]+', big_data.lower())
word_dict = collections.defaultdict(lambda: 0)
for word in words_:
    word_dict[word] += 1


def sentence_to_wordlist(raw_sentence):
    clean_sentence = re.sub("[^a-zA-Z0-9]", " ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)

    return tokens


def tokenize(essay):
    stripped_essay = essay.strip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

    return tokenized_sentences


def tokenize(essay):
    stripped_essay = essay.strip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

    return tokenized_sentences


def avg_word_len(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return sum(len(word) for word in words) / len(words)


def word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return len(words)


def char_count(essay):
    clean_essay = re.sub(r'\s', '', str(essay).lower())

    return len(clean_essay)


def sent_count(essay):
    sentences = nltk.sent_tokenize(essay)

    return len(sentences)


# def count_lemmas_new(essay):
#     lemms = set()
#     tokens = nlp(essay)
#     for token in tokens:
#         if not token.is_oov and not token.is_stop and not token.is_punct:
#             lem_list = lemmatizer(token.text, token.pos_)
#             for lem in lem_list:
#                 lemms.add(lem)
#     return len(lemms)


def count_lemmas(essay):
    tokenized_sentences = tokenize(essay)

    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    lemma_count = len(set(lemmas))

    return lemma_count


def count_spell_error(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    # big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg
    #         and lists of most frequent words from Wiktionary and the British National Corpus.
    #         It contains about a million words.


    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()

    for word in words:
        if not word in word_dict:
            mispell_count += 1

    return mispell_count


def count_pos(essay):
    tokenized_sentences = tokenize(essay)

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1

    return noun_count, adj_count, verb_count, adv_count


# def get_count_vectors_new(essays, max_features=10000):
#     count_vectors = np.zeros(max_features)
#     for essay in essays:
#         tokens = nlp(essay)
#         for token in tokens:
#             if not token.is_oov and not token.is_stop and not token.is_punct:
#                 text = token.text
#                 lemm = lemmatizer(text, token.pos_)
#                 if lemm != token.text:
#                     text = lemm
#                 if text not in feature_words and len(feature_words) < max_features:
#                     feature_words.append(text)
#                 for i in range(len(feature_words)):
#                     if text == feature_words[i]:
#                         count_vectors[i] += 1
#     return feature_words, count_vectors


def get_count_vectors(essays):

    vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 1), stop_words='english')

    count_vectors = vectorizer.fit_transform(essays)

    feature_names = vectorizer.get_feature_names()

    return feature_names, count_vectors


def get_existing_count_vector(essays, feature_names, max_features=10000):
    vectorizer = np.zeros((len(essays), max_features))
    for essay in essays:
        tokens = tokenize(essay)
        for token in tokens:
            for i in range(len(feature_names)):
                if token == feature_names[i]:
                    vectorizer[i] += 1
                    break

    return vectorizer


def extract_features(data):
    print('extracting features')
    features = data.copy()

    print('extracting char count')
    features['char_count'] = features['essay'].apply(char_count)

    print('extracting word count')
    features['word_count'] = features['essay'].apply(word_count)

    print('extracting sentence count')
    features['sent_count'] = features['essay'].apply(sent_count)

    print('extracting avg word length')
    features['avg_word_len'] = features['essay'].apply(avg_word_len)

    print('extracting lemma count')
    features['lemma_count'] = features['essay'].apply(count_lemmas)

    print('check spell errors')
    features['spell_err_count'] = features['essay'].apply(count_spell_error)

    print('extracting pos')
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(
        *features['essay'].map(count_pos))
    print('done')
    return features


def get_features_from_text(text):
    data = [{
        'essay_set': 1,
        'essay': text,
        'domain1_score': 1
    }]
    data = pd.DataFrame.from_dict(data)
    if os.path.exists('feature_names_cv.pkl'):
        print('loading feature names')
        feature_names_cv = load_data('feature_names_cv.pkl')
        X_cv = get_existing_count_vector(data[data['essay_set'] == 1]['essay'], feature_names_cv)
    else:
        feature_names_cv, count_vectors = get_count_vectors(data[data['essay_set'] == 1]['essay'])
        dump_data('feature_names_cv.pkl', feature_names_cv)
        X_cv = count_vectors.toarray()

    features_set1 = extract_features(data[data['essay_set'] == 1])
    X = np.concatenate((features_set1.iloc[:, 3:].as_matrix(), X_cv), axis=1)
    return X