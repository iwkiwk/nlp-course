import os
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.callbacks import Callback
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'


# Load trained w2v model
def load_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model


# Add more training data to trained w2v model
def add_more_train_data(model, data_list):
    sen_list = []
    for line in data_list:
        sen_list.append(line.split())
    model.train(sentences=sen_list, total_examples=len(sen_list), epochs=1)


def process_tags(mat):
    r = mat.shape[0]
    c = mat.shape[1]
    ret = np.zeros((r, c * 4))
    for i in range(r):
        for j in range(c):
            ret[i, 4 * j + 1 - mat[i, j]] = 1
    return ret


def restore_tags(mat):
    r = mat.shape[0]
    c = mat.shape[1] // 4
    ret = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            lt = list(mat[i, 4 * j:4 * j + 4])
            maxind = lt.index(max(lt))
            ret[i, j] = 1 - maxind
    return ret


train_data = pd.read_csv('trainset.csv')
test_data = pd.read_csv('testset.csv')
validate_data = pd.read_csv('validationset.csv')

w2v_model = load_w2v_model('../../input/w2v/w2v.model')
add_more_train_data(w2v_model, train_data['content'])

X_train = train_data['content'].values
y_train = train_data.iloc[:, 2:22].values
X_validation = validate_data['content'].values
y_validation = validate_data.iloc[:, 2:22].values
X_test = test_data['content'].values
y_train_new = process_tags(y_train)
y_validation_new = process_tags(y_validation)

max_features = 30000
maxlen = 100
embed_size = 100

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test) + list(X_validation))
X_train = tokenizer.texts_to_sequences(X_train)
X_validation = tokenizer.texts_to_sequences(X_validation)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_validation = sequence.pad_sequences(X_validation, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    try:
        embedding_vector = w2v_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except:
        pass


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(80, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = get_model()

batch_size = 32
epochs = 1

RocAuc = RocAucEvaluation(validation_data=(x_validation, y_validation_new), interval=1)

hist = model.fit(x_train, y_train_new, batch_size=batch_size, epochs=epochs,
                 validation_data=(x_validation, y_validation_new),
                 callbacks=[RocAuc], verbose=2)

y_pred = model.predict(x_test, batch_size=1024)
result = restore_tags(y_pred)
test_data.iloc[:, 2:22] = result.astype(int)
test_data.to_csv('test_result.csv', index=False)
