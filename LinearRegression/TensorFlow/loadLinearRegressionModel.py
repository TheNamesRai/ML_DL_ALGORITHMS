import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


n_samples = 500
learning_rate = 0.01
epochs = 100


bottle = pd.read_csv('../bottle.csv')
bottle = bottle[['Salnty', 'T_degC']]
bottle.columns = ['Sal', 'Temp']
bottle = bottle.dropna()
bottle = bottle[:][:n_samples]

train_x, train_y = bottle['Temp'], bottle['Sal']



X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# should use np.random.randn() . Using fixed values after looking at the data.
W = tf.Variable(-0.5, tf.float32)
B = tf.Variable(35.0, tf.float32)
pred = tf.add(tf.multiply(X, W), B)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:
	sess.run(init)

	saver.restore(sess, "tmp/model.ckpt")
	output = []
	for x, y in zip(train_x, train_y):
		pred_y = sess.run(pred , feed_dict = {X : x})
		output.append(pred_y)

plt.scatter(train_x, train_y, color='b')
plt.plot(train_x, output, color='k')
plt.show()
