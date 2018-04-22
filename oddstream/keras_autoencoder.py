import keras
from keras.callbacks import History, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping

class KerasAutoEncoder():
    def __init__(self, config):
        self.nIn = config['nIn']
        self.dropout_rate = config['dropout_rate']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']
        self.create_model()


    def create_model(self):
        encoder_input = keras.Input((self.nIn, 1))
        encoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=128, \
                                                         batch_input_shape=(1, None, self.nIn), \
                                                         return_sequences=True, \
                                                         stateful=False, \
                                                         kernel_initializer='glorot_uniform')(encoder_input)
        encoder = keras.layers.Dropout(self.dropout_rate)(encoder)
        encoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=64, \
                                                         batch_input_shape=(1, None, self.nIn), \
                                                         return_sequences=True, \
                                                         stateful=False, \
                                                         return_state=True, \
                                                         kernel_initializer='glorot_uniform')(encoder)
        encoder = keras.layers.Dropout(self.dropout_rate)(encoder)
        encoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=32, \
                                                         batch_input_shape=(1, None, self.nIn), \
                                                         return_sequences=True, \
                                                         stateful=False, \
                                                         return_state=True, \
                                                         kernel_initializer='glorot_uniform')(encoder)

        decoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=32, \
                                                         return_sequences=True, \
                                                         stateful=False, \
                                                         return_state=False, \
                                                         kernel_initializer='glorot_uniform')(encoder)
        decoder = keras.layers.Dropout(self.dropout_rate)(decoder)
        decoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=64, \
                                                         return_sequences=True, \
                                                         stateful=False, \
                                                         return_state=False, \
                                                         kernel_initializer='glorot_uniform')(decoder)
        decoder = keras.layers.Dropout(self.dropout_rate)(decoder)
        decoder = keras.layers.cudnn_recurrent.CuDNNLSTM(units=128, \
                                                         return_sequences=False, \
                                                         stateful=False, \
                                                         return_state=False, \
                                                         kernel_initializer='glorot_uniform')(decoder)
        decoder_output = keras.layers.Dense(units=self.nIn, activation="tanh")(decoder)
        self.ae_model = keras.Model(encoder_input, decoder_output)
        self.ae_model.compile(loss="mse", optimizer=keras.optimizers.Adam())
        self.encoder_model = keras.Model(encoder_input, encoder)


    def train(self, X_train):
        history = History()
        nan_terminator = TerminateOnNaN()
        early_stopping = EarlyStopping(patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='loss', \
                                      factor=0.5, \
                                      patience=50, \
                                      verbose=1, \
                                      mode='auto', \
                                      epsilon=0.0001, \
                                      cooldown=0, \
                                      min_lr=1e-8)

        self.ae_model.fit(x=X_train, \
                          y=X_train, \
                          epochs=self.epochs, \
                          verbose=1, \
                          batch_size=self.batch_size, \
                          validation_split=self.validation_split, \
                          shuffle=True, \
                          callbacks=[history, early_stopping, nan_terminator, reduce_lr])


    def extract_features(self, X_test):
        encoded = self.encoder_model.predict_on_batch(X_test)
        # TODO post-process data