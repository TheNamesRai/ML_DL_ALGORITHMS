## Getting around 95% accuracy


import tensorflow as tf 

#loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#data preprocessing 
#flattening and normalization
x_train = x_train.reshape(60000,784) / 255
x_test = x_test.reshape(10000, 784) / 255

#one hot 
with tf.Session() as sess:
	y_train = sess.run(tf.one_hot(y_train,10))
	y_test = sess.run(tf.one_hot(y_test,10))


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
learning_rate = 0.01
epochs = 20
batch_size = 100
batches = int(x_train.shape[0] / batch_size)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
						'biases' : tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) # default learning rate = 0.01

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epoch_loss = 0
			for i in range(batches):
				offset = i * epoch
				x_batch = x_train[offset: offset + batch_size]
				y_batch = y_train[offset: offset + batch_size]
				#epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer , cost], feed_dict = {x:x_batch , y:y_batch})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)


		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))

train_neural_network(x)