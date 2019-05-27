import os
import numpy as np
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#STEP 2 - Architecture selection
def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[2, 2], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('flat'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
    l_layer = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(l_layer, name='ybar')
    if logits: 
        return y, l_layer
    return y 

#FGSM
def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0, clip_max=1):
    xadv = tf.identity(x)
    ybar = model(xadv)
    yshape = ybar.get_shape().as_list() 
    ydim = yshape[1]
    indices = tf.argmax(ybar, axis=1) 
    target = tf.cond(
     tf.equal(ydim,1),
     lambda: tf.nn.relu(tf.sign(ybar-0.5)),
     lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))
    loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
    noise_fn = tf.sign
    eps = tf.abs(eps)
    def cond(xadv, i):
        return tf.less(i, epochs)
    def body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1
    #STEP 3 - Labeling
    xadv, _ = tf.while_loop(cond, body, (xadv, 0), back_prop=False, name='fast_gradient')
    return xadv

class Environment():
	pass

ambiente = Environment()

with tf.variable_scope('model'):
    ambiente.x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    ambiente.y = tf.placeholder(tf.float32, (None, 10), name='y')
    ambiente.ybar, logits = model(ambiente.x, logits=True)
    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(ambiente.y, axis=1), tf.argmax(ambiente.ybar, axis=1))
        ambiente.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ambiente.y, logits=logits)
        ambiente.loss = tf.reduce_mean(cross_entropy, name='loss')
    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        ambiente.train_op = optimizer.minimize(ambiente.loss)
with tf.variable_scope('model', reuse=True):
    ambiente.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    ambiente.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    ambiente.x_fgsm = fgm(model, ambiente.x, epochs=ambiente.fgsm_epochs, eps=ambiente.fgsm_eps)

#STEP 4 - Training
def training(sess, ambiente, X_data, Y_data, X_valid=None, y_valid=None, shuffle=True, batch=128, epochs=1):
	Xshape = X_data.shape
	n_data = Xshape[0]
	n_batches = int(n_data/batch)
	print(X_data.shape)
	for ep in range(epochs):
		print('epoch number: ', ep+1)
		if shuffle:
			ind = np.arange(n_data)
			np.random.shuffle(ind)
			X_data = X_data[ind]
			Y_data = Y_data[ind]
		for i in range(n_batches):
			print(' batch {0}/{1}'.format(i + 1, n_batches), end='\r')
			start = i*batch 
			end = min(start+batch, n_data)
			sess.run(ambiente.train_op, feed_dict={ambiente.x: X_data[start:end], ambiente.y: Y_data[start:end]})
		evaluate(sess, ambiente, X_valid, y_valid)
		
def evaluate(sess, ambiente, X_test, Y_test, batch=128):
	n_data = X_test.shape[0]
	n_batches = int(n_data/batch)
	totalAcc = 0
	totalLoss = 0
	for i in range(n_batches):
		print(' batch {0}/{1}'.format(i + 1, n_batches), end='\r')
		start = i*batch 
		end = min(start+batch, n_data)
		batch_X = X_test[start:end]
		batch_Y = Y_test[start:end]
		batch_loss, batch_acc = sess.run([ambiente.loss, ambiente.acc], feed_dict={ambiente.x: batch_X, ambiente.y: batch_Y})
		totalAcc = totalAcc + batch_acc*(end-start)
		totalLoss = totalLoss + batch_loss*(end-start)
	totalAcc = totalAcc/n_data
	totalLoss = totalLoss/n_data
	print('acc: {0:.3f} loss: {1:.3f}'.format(totalAcc, totalLoss))
	return totalAcc, totalLoss

#STEP 5 - Dataset Augmentation
def perform_fgsm(sess, ambiente, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nInizio FGSM')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(ambiente.x_fgsm, feed_dict={
            ambiente.x: X_data[start:end],
            ambiente.fgsm_eps: eps,
            ambiente.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()
    return X_adv


def main():
    #STEP 1 - Initial Dataset Collection
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels 
    tf.logging.set_verbosity(old_v)
    i = int(X_train.shape[0] * 0.9)
    X_validate = X_train[i:]
    X_train = X_train[:i]
    y_validate = y_train[i:]
    y_train = y_train[:i]
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    training(sess, ambiente, X_train, y_train, X_validate, y_validate, shuffle=False, batch=128, epochs=5)
    evaluate(sess, ambiente, X_test, y_test)
    X_adv = perform_fgsm(sess, ambiente, X_test, eps=0.02, epochs=12)
    evaluate(sess, ambiente, X_adv, y_test)

if __name__ == "__main__":
    main()
