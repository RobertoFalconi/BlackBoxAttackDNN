import os
import numpy as np
import tensorflow as tf
import math

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("MNIST_data", one_hot=True, reshape=False, validation_size=0)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
factor = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[factor:]
X_train = X_train[:factor]
y_valid = y_train[factor:]
y_train = y_train[:factor]

n_classes = 10


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


####################################################################################
def jsma(model, x, y=None, epochs=1, eps=1.0, k=1, clip_min=0.0, clip_max=1.0, score_fn=lambda t, o: t * tf.abs(o)):

    n = tf.shape(x)[0] # how many images are input? n = 128 by batch size 

    target = tf.cond(tf.equal(0, tf.rank(y)), # tf.rank(y) returns rank of a tensor y
                     lambda: tf.zeros([n], dtype=tf.int32) + y,
                     lambda: y)

    target = tf.stack((tf.range(n), target), axis=1) # 2xn

    """
    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
    """

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

    def _cond(i, xadv):
        return tf.less(i, epochs)

    def _body(i, xadv):
        ybar = model(xadv)

        dy_dx = tf.gradients(ybar, xadv)[0] # Nx28x28x1


        # gradients of target w.r.t input
        yt = tf.gather_nd(ybar, yind) #yind = target = (2 x n) = random labels assigned by np.random function
        dt_dx = tf.gradients(yt, xadv)[0]   #[0] makes it list then you can do do_dx = dy_dx - dt_dx # Nx28x28x1

        # gradients of non-targets w.r.t input
        do_dx = dy_dx - dt_dx # Nx28x28x1

        c0 = tf.logical_or(eps < 0, xadv < clip_max) # returns true when either of these two is true
        c1 = tf.logical_or(eps > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond) #tf.to_float([1,2,3]) produces just [1.,2.,3.] # return 0 or return 1

        # saliency score for each pixel
        score = cond * score_fn(dt_dx, do_dx) # function to calculate the saliency score for each pixel

        shape = score.get_shape().as_list() # 784 x 1 matrix # To get the shape as a list of ints, do tensor.get_shape().as_list()
        dim = _prod(shape[1:]) # multiplication except the first one
        score = tf.reshape(score, [-1, dim]) # make 1D matrix


        # find the pixel with the highest saliency score
        ind = tf.argmax(score, axis=1) # find the pixel with highest value
        dx = tf.one_hot(ind, dim, on_value=eps, off_value=0.0)
        dx = tf.reshape(dx, [-1] + shape[1:])

        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return i+1, xadv

    _, xadv = tf.while_loop(_cond, _body, (0, tf.identity(x)), back_prop=False, name='_jsma_batch')

    return xadv

############################################################################


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, 28, 28, 1),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, 10), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.target = tf.placeholder(tf.int32, (), name='target')
    env.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
    env.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
    env.x_jsma = jsma(model, env.x, env.target, eps=env.adv_eps, epochs=env.adv_epochs)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
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
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))



def make_jsma(sess, env, X_data, epochs=0.2, eps=1.0, batch_size=128):
    """
    Generate JSMA by running env.x_jsma.
    """
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.target: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, epochs=2)

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)

print('\nGenerating adversarial data')

X_adv = make_jsma(sess, env, X_test, epochs=40, eps=0.8)

print('\nEvaluating on adversarial data')

evaluate(sess, env, X_adv, y_test)
