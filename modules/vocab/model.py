from keras.models import load_model
from modules.vocab.feature_extraction import get_features_from_text, get_synonyms
import numpy as np


class Model:
    def __init__(self):
        self.model_path = 'modules/vocab/weights.hdf5'
        self.weight_path = 'modules/vocab/model.hdf5'
        self.model = load_model(self.model_path)
        self.model.load_weights(self.weight_path)
        self.bands = np.arange(4.5, 9.5, 0.5)
        self.model._make_predict_function()

    def _convert_to_band(self, est_score):
        est_score = (est_score + 1) * 4.5
        for i in range(len(self.bands)):
            if est_score > self.bands[i]:
                continue
            else:
                lesser_band = self.bands[max(i - 1, 0)]
                if est_score - lesser_band <= 0.25:
                    return lesser_band
                return self.bands[i]
        return self.bands[-1]

    def predict(self, text):
        if text is None:
            return 0
        features = get_features_from_text(text)
        predict_norm_score = self.model.predict(features)
        predicted_score = self._convert_to_band(predict_norm_score)
        list_synonyms = get_synonyms(text)
        vocab = dict(vocab_recommend=list_synonyms, predicted_score=predicted_score)
        return vocab
