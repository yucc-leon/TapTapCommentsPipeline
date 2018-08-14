import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
from datautils import *
from sklearn.metrics import precision_score,recall_score,f1_score

def score(pred, label, gate=0.5):
    if len(label.shape) == 1:
        p = (pred>gate).astype("int")
        p = np.squeeze(p)
        l = label
    else:
        p = np.argmax(pred, axis=1)
        l = np.argmax(label, axis=1)
    pre_score = precision_score(l, p)
    rec_score = recall_score(l, p)
    f_score = f1_score(l, p)
    return pre_score, rec_score, f_score


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
             (input_shape[-1],),
             initializer=self.init,
             name='context',
             regularizer=self.W_regularizer,
             constraint=self.W_constraint
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
def conv_block(data, convs = [3,4,5], f = 128, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f,kernel_size=c,padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

def get_textcnn(seq_length, embed_weight, class_nb):
    content = Input(shape=(seq_length,),dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0],weights=[embed_weight],output_dim=embed_weight.shape[1],trainable=False)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    feat = conv_block(trans_content)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(class_nb,activation="softmax")(fc)
    model = Model(inputs=content,outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def rnn_block(data, unit_nb=256, name='rnn_block', drop_p=.2):
    rnn = Bidirectional(GRU(unit_nb, return_sequences=True)(data))
    droprnn = Dropout(drop_p)(rnn)
    return droprnn

def get_rnn(seq_length, embed_weight, class_nb):
    content = Input(shape=(seq_length, ), dtype='int32')
    embedding = Embedding(input_dim=embed_weight.shape[0],weights=[embed_weight],output_dim=embed_weight.shape[1],trainable=False)
    trans_content = Activation(activation='relu')(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    rnndrop = rnn_block(trans_content)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(rnndrop)))
    output = Dense(class_nb,activation="softmax")(fc)
    model = Model(inputs=content,outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_han(sen_nb, per_sen_len, embed_weight, class_nb, mask_zero=False):
    sentence_input = Input(shape=(per_sen_len,), dtype="int32")
    embedding = Embedding(
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(per_sen_len)(word_bigru)
    sent_encode = Model(sentence_input, word_attention)

    review_input = Input(shape=(sen_nb, per_sen_len), dtype="int32")
    review_encode = TimeDistributed(sent_encode)(review_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(review_encode)
    sent_attention = Attention(sen_nb)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(sent_attention)))
    output = Dense(class_nb,activation="softmax")(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model, sent_encode

def convs_block_v2(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Conv1D(f, c, activation='elu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(f//2, 2, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    return concatenate(pools, name=name)

def get_textcnn_v2(seq_length, embed_weight, class_nb):
    content = Input(shape=(seq_length,),dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0],weights=[embed_weight],output_dim=embed_weight.shape[1],name="embedding",trainable=False)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    unvec = conv_block(trans_content)
    dropfeat = Dropout(0.4)(unvec)
    fc = Dropout(0.4)(Activation(activation="relu")(BatchNormalization()(Dense(300)(dropfeat))))
    output = Dense(class_nb,activation="softmax")(fc)
    model = Model(inputs=content,outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model