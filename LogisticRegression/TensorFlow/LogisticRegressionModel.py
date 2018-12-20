import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

#loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Hyperparameters
learning_rate = 0.01
epochs = 20
batch_size = 100
batches = int(x_train.shape[0] / batch_size)


#data preprocessing 
#flattening and normalization
x_train = x_train.reshape(60000,784) / 255
x_test = x_test.reshape(10000, 784) / 255

#one hot 
with tf.Session() as sess:
	y_train = sess.run(tf.one_hot(y_train,10))
	y_test = sess.run(tf.one_hot(y_test,10))


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(0.1*np.random.randn(784,10).astype(np.float32))
B = tf.Variable(0.1*np.random.randn(10).astype(np.float32))

pred = tf.add(tf.matmul(X,W), B)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for i in range(batches):
			offset = i * epoch
			x = x_train[offset: offset + batch_size]
			y = y_train[offset: offset + batch_size]
			sess.run(optimizer, feed_dict={X : x, Y : y})
			c = sess.run(cost,feed_dict={X : x, Y : y})

		if not epoch % 1:
			print(f'epoch : {epoch}, cost : {c}')
			
	saver.save(sess, 'tmp/model.ckpt')

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	acc = accuracy.eval({X:x_test, Y:y_test})
	print(acc)
