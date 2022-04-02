# Import libraries
import numpy as np
import pickle
from transformers import TFBertModel
from transformers import BertTokenizer
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras import Model

import subprocess


gpu_memory_info = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv", shell=True)

gpus = {}
num = '0123456789'
g_num = 0
g_mem = ''
mess = str(gpu_memory_info)
for i in range(len(mess)):
  if mess[i] in num: 
    g_mem += mess[i]
    if mess[i+1] not in num:
      gpus[g_num] = g_mem
      g_num += 1
      g_mem = ''

use_gpu = 0
gpu_mem = 0
for i in gpus.keys():
  if gpu_mem < int(gpus[i]):
    gpu_mem = int(gpus[i])
    use_gpu = i

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/device:GPU:' + str(use_gpu)):
    # Bert classifer model
    class BERT_MODEL():
        ''' info '''

        # Class initialization
        def __init__(self):

            with open('model/df_columns', 'rb') as f: 
                self.df_columns = pickle.load(f)
            with open('model/c_weight', 'rb') as f: 
                self.c_weight = pickle.load(f)            

            self.MAX_SEQ_LEN = 256
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

            output = Dense(len(self.c_weight), activation="sigmoid")(avg)

            model = Model(inputs = inputs, outputs = output)
            model.compile()
            model.load_weights('model/model.h5')

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
            print(np.round(pred[0]).astype(int))
            print(pred[0])
            for i in self.convert_predict(np.round(pred[0]).astype(int)):
                if len(prediction) != 0:
                    prediction = prediction + '   |   '
                prediction = prediction + i
            return prediction
