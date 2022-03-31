# Import libraries
import numpy as np
import pickle
from transformers import TFBertModel
from transformers import BertTokenizer
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.metrics import AUC, CategoricalAccuracy
from tensorflow.keras import Model

# Bert classifer model
class BERT_MODEL():
    ''' info '''

    # Class initialization
    def __init__(self):

        with open('model/df_columns', 'rb') as f: 
            self.df_columns = pickle.load(f)
        with open('model/c_weight', 'rb') as f: 
            self.c_weight = pickle.load(f)            

        self.MAX_SEQ_LEN = 1024
        self.BERT_NAME = 'DeepPavlov/rubert-base-cased-conversational'
        self.model = self.load_model()
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_NAME)

    
    # Model definition function
    def load_model(self):
        
        input_ids = Input(shape = (self.MAX_SEQ_LEN,), dtype = tf.int32, name = 'input_ids')
        input_type = Input(shape = (self.MAX_SEQ_LEN,), dtype = tf.int32, name = 'token_type_ids')
        input_mask = Input(shape = (self.MAX_SEQ_LEN,), dtype = tf.int32, name = 'attention_mask')

        inputs = [input_ids, input_type, input_mask]

        bert = TFBertModel.from_pretrained(self.BERT_NAME, from_pt=True)
        bert_outputs = bert(inputs)

        last_hidden_states = bert_outputs.last_hidden_state
        avg = GlobalAveragePooling1D()(last_hidden_states)
        avg = Dropout(0.4)(avg)

        output = Dense(len(self.c_weight), activation="softmax")(avg)
        model = Model(inputs = inputs, outputs = output)
        opt = tfa.optimizers.RectifiedAdam(learning_rate = 1e-6)
        loss = 'categorical_crossentropy'

        model.compile(loss = loss, optimizer = opt, metrics = [AUC(multi_label = True, curve = "ROC"), CategoricalAccuracy()])

        model.load_weights("model/rubert_model.h5")
        print(model.summary())

        return model

    def convert_predict(self, pred):
        ind = []
        for i in range(len(pred)):
            if pred[i] == 1: ind.append(self.df_columns[i])
        return ind

    # Text tokenization function
    def prepare_bert_input(self, sentences):
        encodings = self.tokenizer(sentences, truncation = True, padding = 'max_length',
                                    max_length=self.MAX_SEQ_LEN)
        encod = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),
                np.array(encodings["attention_mask"])]
        return encod

    # Predict function
    def predict(self, input):
        input = self.prepare_bert_input(input)
        pred = self.model.predict(input)
        prediction = ''
        for i in self.convert_predict(np.round(pred[0]).astype(int)):
            if len(prediction) != 0:
                prediction = prediction + '   |   '
            prediction = prediction + i
        return prediction
