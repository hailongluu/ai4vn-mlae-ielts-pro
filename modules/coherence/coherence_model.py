import re
import os
import pickle
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
import contractions
from nltk import tokenize

MAX_SENT_LENGTH = 30
MAX_SENTS = 15
MAX_SENTS_TOPIC = 15
MAX_NB_WORDS = 400
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


# import keras.backend as KTF
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
#
# KTF.set_session(sess)

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim,)), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class CoherenceModel():
    def __init__(self, matrix_path="embedding_matrix.txt", weight_path="pre-35-0.0187.h5", word_path="word_index.pkl"):
        f = open(word_path, 'rb')
        self.word_index = pickle.load(f)
        f.close()
        self.embedding_matrix = np.genfromtxt(matrix_path)
        print("embedding matrix", self.embedding_matrix.shape)
        print('Total %s unique tokens.' % len(self.word_index))
        self.model = self.build_model(weight_path)
        self.model._make_predict_function()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = str(string)
        string = re.sub(r"\n", " ", string)
        string = re.sub(r"-", " ", string)

        string = contractions.fix(string)
        string = string.translate(str.maketrans('', '', "!#$%&’'?()*+-/:<=>@[\]^_`{|}~"))
        return string.strip().lower()

    def load_data(self, txt_essay, txt_topic):
        reviews = []
        texts = []
        topics = []
        txt_essay = self.clean_str(txt_essay)
        txt_topic = self.clean_str(txt_topic)
        texts.append(txt_essay + " " + txt_topic)
        sentences = tokenize.sent_tokenize(txt_essay)
        reviews.append(sentences)
        topics.append(tokenize.sent_tokenize(txt_topic))
        return reviews, texts, topics

    def prepare_data(self, list_sent, texts, max_sents):
        data = np.zeros((len(texts), max_sents, MAX_SENT_LENGTH), dtype='int32')
        # ----text to index essay-------------
        for i, sentences in enumerate(list_sent):
            for j, sent in enumerate(sentences):
                if j < max_sents:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        try:
                            if k < MAX_SENT_LENGTH and self.word_index[word] < MAX_NB_WORDS:
                                data[i, j, k] = self.word_index[word]
                                k = k + 1
                        except KeyError:
                            continue
        return data

    def build_model(self, weight):
        # building Hierachical Attention network
        embedding_layer = Embedding(self.embedding_matrix.shape[0],
                                    EMBEDDING_DIM,
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True,
                                    mask_zero=True)

        # ----Essay model--------
        sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        l_lstm = Dropout(0.5)(l_lstm)
        l_att = AttLayer(50)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(50, return_sequences=True))(review_encoder)
        l_lstm_sent = Dropout(0.5)(l_lstm_sent)
        l_att_sent = AttLayer(50)(l_lstm_sent)

        # ----Topic model--------
        sentence_input_2 = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sentence_input_2)
        l_lstm_2 = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences_2)
        l_lstm_2 = Dropout(0.3)(l_lstm_2)
        l_att_2 = AttLayer(50)(l_lstm_2)
        topicSentEncoder = Model(sentence_input_2, l_att_2)

        topic_input = Input(shape=(MAX_SENTS_TOPIC, MAX_SENT_LENGTH), dtype='int32')
        topic_encoder = TimeDistributed(topicSentEncoder)(topic_input)
        l_lstm_sent_topic = Bidirectional(GRU(50, return_sequences=True))(topic_encoder)
        l_lstm_sent_topic = Dropout(0.3)(l_lstm_sent_topic)
        l_att_sent_topic = AttLayer(50)(l_lstm_sent_topic)

        # ----Concat------
        pe = Multiply()([l_att_sent, l_att_sent_topic])
        feat_vec = Concatenate(axis=1)([l_att_sent, pe])
        preds = Dense(1, activation='sigmoid')(feat_vec)
        model = Model([review_input, topic_input], preds)

        model.load_weights(weight)
        return model

    def predict(self, essay, topic):
        essay, text, topic = self.load_data(essay, topic)
        essay = self.prepare_data(essay, text, MAX_SENTS)
        topic = self.prepare_data(topic, text, MAX_SENTS_TOPIC)
        pred_score = self.model.predict([essay, topic])[0][0] * 9
        decimal = pred_score - int(pred_score)
        if decimal >= 0 and decimal <= 0.25:
            decimal = 0
        elif decimal > 0.25 and decimal <= 0.75:
            decimal = 0.5
        else:
            decimal = 1
        pred_score = int(pred_score) + decimal
        return pred_score


# model = CoherenceModel()
# print(model.predict(
#     "It is preferred to live in a house by some people but living in an apartment is considered more advantageous by the other group of people. I strongly agree with the second group of residents who prefer to live in an apartment rather being in a house. Living in an apartment makes the family closely-bonded as there are not multiple floors. The family members spend more time with each other while living on the same floor as compared to living in different floors in a house. Secondly, the safety in an apartment is more as the apartment’s society is always deployed with ample number of security guards along with CCTV cameras at all floors, entrance and staircase the society events and competitions give a platform to their children to interact with kids from different religions and background. It also encourage them to perform in front of large audience which enhances their public speaking skills. In total, living in an apartment is comparatively has more plus points than residing in a house.",
#     "Some people prefer living in an apartment while others believe living in a house brings more advantages. Which, out of the two, is better?"))
