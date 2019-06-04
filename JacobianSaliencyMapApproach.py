import os
import numpy as np
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CLASSES = 10

#STEP 2 - Architecture Selection
def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[2, 2], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y

def jsma(model, x, y=None, epochs=1, eps=1.0, k=1, clip_min=0.0, clip_max=1.0, score_fn=lambda t, o: t * tf.abs(o)):
    n = tf.shape(x)[0]
    target = tf.cond(tf.equal(0, tf.rank(y)), lambda: tf.zeros([n], dtype=tf.int32) + y, lambda: y)
    target = tf.stack((tf.range(n), target), axis=1) # 2xn
    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))
    return _jsma_impl(model, x, target, epochs=epochs, eps=eps, clip_min=clip_min, clip_max=clip_max, score_fn=score_fn)

def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret

def _jsma_impl(model, x, yind, epochs, eps, clip_min, clip_max, score_fn):
    def cond(i, xadv):
        return tf.less(i, epochs)
    def body(i, xadv):
        ybar = model(xadv)
        dy_dx = tf.gradients(ybar, xadv)[0]   
        yt = tf.gather_nd(ybar, yind) 
        dt_dx = tf.gradients(yt, xadv)[0]        
        do_dx = dy_dx - dt_dx
        c0 = tf.logical_or(eps < 0, xadv < clip_max) 
        c1 = tf.logical_or(eps > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond)        
        score = cond * score_fn(dt_dx, do_dx) 
        shape = score.get_shape().as_list() 
        dim = _prod(shape[1:]) 
        score = tf.reshape(score, [-1, dim])
        ind = tf.argmax(score, axis=1)
        dx = tf.one_hot(ind, dim, on_value=eps, off_value=0.0)
        dx = tf.reshape(dx, [-1] + shape[1:])
        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return i+1, xadv
    #STEP 3 - Substitute Dataset Labeling
    _, xadv = tf.while_loop(cond, body, (0, tf.identity(x)), back_prop=False, name='_jsma_batch')
    return xadv

class Environment():
    pass

# CLASS ENVIRONMENT DEFINITION, BEFORE RUNNING MAIN
ambiente =  Environment()

with tf.variable_scope('model'):
    ambiente.x = tf.placeholder(tf.float32, (None, 28, 28, 1),name='x')
    ambiente.y = tf.placeholder(tf.float32, (None, 10), name='y')
    ambiente.training = tf.placeholder_with_default(False, (), name='mode')
    ambiente.ybar, logits = model(ambiente.x, logits=True, training=ambiente.training)
    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(ambiente.y, axis=1), tf.argmax(ambiente.ybar, axis=1))
        ambiente.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ambiente.y, logits=logits)
        ambiente.loss = tf.reduce_mean(xent, name='loss')
    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        ambiente.train_op = optimizer.minimize(ambiente.loss)
    ambiente.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    ambiente.target = tf.placeholder(tf.int32, (), name='target')
    ambiente.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
    ambiente.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
    ambiente.x_jsma = jsma(model, ambiente.x, ambiente.target, eps=ambiente.adv_eps, epochs=ambiente.adv_epochs)

def evaluate(sess, ambiente, X_data, y_data, batch_size=128):
    print('\nValutazione')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run([ambiente.loss, ambiente.acc],feed_dict={ambiente.x: X_data[start:end], ambiente.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample
    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

#STEP 4 - Substitute DNN F Training
def train(sess, ambiente, X_data, y_data, X_valid=None, y_valid=None, epochs=1, load=False, shuffle=True, batch_size=128, name='model'): 
    if load:
        if not hasattr(ambiente, 'saver'):
            return print('\nError')
        return ambiente.saver.restore(sess, 'model/{}'.format(name))
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))
        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(ambiente.train_op, feed_dict={ambiente.x: X_data[start:end], ambiente.y: y_data[start:end], ambiente.training: True})
        if X_valid is not None:
            evaluate(sess, ambiente, X_valid, y_valid)
    if hasattr(ambiente, 'saver'):
        os.makedirs('model', exist_ok=True)
        ambiente.saver.save(sess, 'model/{}'.format(name))

#STEP 5 - Jacobian-Based Dataset Augmentation
def make_jsma(sess, ambiente, X_data, epochs=0.2, eps=1.0, batch_size=128):
    print('\nInizio JSMA')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {ambiente.x: X_data[start:end], ambiente.target: np.random.choice(CLASSES),ambiente.adv_epochs: epochs,ambiente.adv_eps: eps}
        adv = sess.run(ambiente.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()
    return X_adv


def main():
#STEP 1 - Substitute Training Dataset Collection
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
# read images from dataset
    mnist = mnist_data.read_data_sets("MNIST_data", one_hot=True, reshape=False, validation_size=0)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    tf.logging.set_verbosity(old_v)
# 90% of dataset is training set, 10% is validation set
    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]
    factor = int(X_train.shape[0] * 0.9)
    X_valid = X_train[factor:]
    X_train = X_train[:factor]
    y_valid = y_train[factor:]
    y_train = y_train[:factor]
# start tensorflow session
# runs STEP 2
    print('\nInizializzazione grafo')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
# runs training and evaluating
# STEP 4
    train(sess, ambiente, X_train, y_train, X_valid, y_valid, epochs=2)
    print('\nValutazione su dati nuovi')
    evaluate(sess, ambiente, X_test, y_test)
    print('\nGenerazione dati avversari')
    X_adv = make_jsma(sess, ambiente, X_test, epochs=40, eps=0.8)
    print('\nValutazione su dati avversari')
    evaluate(sess, ambiente, X_adv, y_test)

if __name__ == "__main__":
    main()

# MAIN:
# STEP 1
# DATASET COLLECTION
# STEP 2
# INTERACTIVE SESSION -> ENVIRONMENT:
# MODEL
# FGM (ADVERSARIAL MODEL)
# STEP 3
# LABELING 
# STEP 4
# TRAINING 
# EVALUATE
# PERFORM_FGSM
# EVALUATE
# STEP 5
# AUGMENTATION
