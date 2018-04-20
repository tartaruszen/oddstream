import tensorflow as tf
import numpy as np
from .autoencoder_model import AutoEncoderModel

def train_ae_model(X_train, X_val, config, plot_every = 100, max_iterations = 1000):
    n = X_train.shape[0]
    d = X_train.shape[1]

    n_val = X_val.shape[0]

    batch_size = config['batch_size']
    dropout = config['dropout']

    config['sl'] = sl = d
    epochs = np.floor(config['batch_size']  * max_iterations / n)
    print('Train with approximately %d epochs' % epochs)

    ae_model = AutoEncoderModel(config)
    sess = tf.Session()
    perf_collect = np.zeros((2, int(np.floor(max_iterations / plot_every))))
    sess.run(ae_model.init_op)
    step = 0  # Step is a counter for filling the numpy array perf_collect
    for i in range(max_iterations):
        batch_ind = np.random.choice(n, batch_size, replace=False)
        result = sess.run([ae_model.loss, ae_model.loss_seq, ae_model.loss_lat_batch, ae_model.train_step],
                          feed_dict={ae_model.x: X_train[batch_ind], ae_model.keep_prob: dropout})

        if i % plot_every == 0:
            # Save train performances
            perf_collect[0, step] = loss_train = result[0]
            loss_train_seq, lost_train_lat = result[1], result[2]

            # Calculate and save validation performance
            batch_ind_val = np.random.choice(n_val, batch_size, replace=False)

            result = sess.run([ae_model.loss, ae_model.loss_seq, ae_model.loss_lat_batch, ae_model.merged],
                              feed_dict={ae_model.x: X_val[batch_ind_val], ae_model.keep_prob: 1.0})
            perf_collect[1, step] = loss_val = result[0]
            loss_val_seq, lost_val_lat = result[1], result[2]
            # and save to Tensorboard
            summary_str = result[3]

            print("At %6s / %6s train (%5.3f, %5.3f, %5.3f), val (%5.3f, %5.3f,%5.3f) in order (total, seq, lat)" % (
                i, max_iterations, loss_train, loss_train_seq, lost_train_lat, loss_val, loss_val_seq, lost_val_lat))
            step += 1
    return ae_model



def extract_latent(X, sess, model, config):
    n = X.shape[0]
    sess = tf.Session()
    start = 0
    z_run = []
    batch_size = config['batch_size']
    while start + batch_size < n:
        run_ind = range(start, start + batch_size)
        z_mu_fetch = sess.run(model.z_mu, feed_dict={model.x: X[run_ind], model.keep_prob: 1.0})
        z_run.append(z_mu_fetch)
        start += batch_size

    z_run = np.concatenate(z_run, axis=0)
    return z_run

