import tensorflow as tf

def train_ae_model(X_train, X_val, config, plot_every = 100, max_iterations = 1000):
    n = X_train.shape[0]

    epochs = np.floor(config['batch_size']  * max_iterations / N)
    print('Train with approximately %d epochs' % epochs)
