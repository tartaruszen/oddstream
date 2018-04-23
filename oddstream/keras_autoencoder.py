import keras
import numpy as np
from keras.callbacks import History, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping

class KerasAutoEncoder():
    def __init__(self, config, callbacks = [], useCUDNN = False):
        self.nIn = config['nIn']
        self.dropout_rate = config['dropout_rate']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']
        self.callbacks = callbacks
        if useCUDNN:
            self.create_model_cudnn()
        else:
            self.create_model()

    def create_model(self):
        encoder_input = keras.Input((self.nIn, 1))
        encoder = keras.layers.LSTM(units=128,
                                    batch_input_shape=(1, None, self.nIn),
                                    return_sequences=True,
                                    stateful=False,
                                    dropout=self.dropout_rate,
                                    kernel_initializer='glorot_uniform')(encoder_input)
        encoder = keras.layers.LSTM(units=64,
                                    batch_input_shape=(1, None, self.nIn),
                                    return_sequences=True,
                                    stateful=False,
                                    return_state=False,
                                    dropout=self.dropout_rate,
                                    kernel_initializer='glorot_uniform')(encoder)
        encoder = keras.layers.LSTM(units=32,
                                    batch_input_shape=(1, None, self.nIn),
                                    return_sequences=True,
                                    stateful=False,
                                    return_state=True,
                                    dropout=self.dropout_rate,
                                    kernel_initializer='glorot_uniform')(encoder)

        decoder = keras.layers.LSTM(units=32,
                                    return_sequences=True,
                                    stateful=False,
                                    return_state=False)(encoder)
        decoder = keras.layers.LSTM(units=64,
                                    return_sequences=True,
                                    stateful=False,
                                    return_state=False,
                                    dropout=self.dropout_rate)(decoder)
        decoder = keras.layers.LSTM(units=128,
                                    return_sequences=False,
                                    stateful=False,
                                    return_state=False,
                                    dropout=self.dropout_rate)(decoder)
        decoder_output = keras.layers.Dense(units=self.nIn, activation="tanh")(decoder)
        self.ae_model = keras.Model(encoder_input, decoder_output)
        self.ae_model.compile(loss="mse", optimizer=keras.optimizers.Adam())
        self.encoder_model = keras.Model(encoder_input, encoder)

    def create_model_cudnn(self):
        encoder_input = keras.Input((self.nIn, 1))
        encoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=128,
                                                         batch_input_shape=(1, None, self.nIn),
                                                         return_sequences=True,
                                                         stateful=False,
                                                         kernel_initializer='glorot_uniform')(encoder_input)
        encoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=64,
                                                         batch_input_shape=(1, None, self.nIn),
                                                         return_sequences=True,
                                                         stateful=False,
                                                         return_state=False,
                                                         kernel_initializer='glorot_uniform')(encoder)
        encoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=32,
                                                         batch_input_shape=(1, None, self.nIn),
                                                         return_sequences=True,
                                                         stateful=False,
                                                         return_state=True,
                                                         kernel_initializer='glorot_uniform')(encoder)

        decoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=32,
                                                         return_sequences=True,
                                                         stateful=False,
                                                         return_state=False)(encoder)
        decoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=64,
                                                         return_sequences=True,
                                                         stateful=False,
                                                         return_state=False)(decoder)
        decoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=128,
                                                         return_sequences=False,
                                                         stateful=False,
                                                         return_state=False)(decoder)
        decoder_output = keras.layers.Dense(units=self.nIn, activation="tanh")(decoder)
        self.ae_model = keras.Model(encoder_input, decoder_output)
        self.ae_model.compile(loss="mse", optimizer=keras.optimizers.Adam())
        self.encoder_model = keras.Model(encoder_input, encoder)

    def train(self, X_train):
        self.ae_model.fit(x=X_train,
                          y=X_train.reshape(X_train.shape[0], self.nIn),
                          epochs=self.epochs,
                          verbose=1,
                          batch_size=self.batch_size,
                          validation_split=self.validation_split,
                          callbacks = self.callbacks,
                          shuffle=True)

    def extract_features(self, X_test):
        encoded = self.encoder_model.predict_on_batch(X_test)[0]
        encoder_features = np.array([])
        for i in range(0, X_test.shape[0]):
            encoder_features = np.append(encoder_features, np.mean(encoded[i, :, :], axis=1))
        return encoder_features.reshape(X_test.shape[0], self.nIn)