#Build Autoencoder
import tensorflow as tf
import numpy as np
import data_loader 
import config

class Autoencoder():

	def __init__(self):
		#self.config = config
		self.data_loader = data_loader
		data_sets = data_loader.load_data()
		self.train_x = data_sets['images_train']
		self.train_y = data['labels_train']

	def model_architecture(self, x, y):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.01)):
			conv1 = slim.conv2d(x, 64, [5, 5], scope='conv1')
			pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
			conv2 = slim.conv2d(pool1, 128, [5, 5], scope='conv2') 
			pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
			fc3 = slim.fully_connected(pool2, 256, activation_fn=tf.nn.relu, scope='fc3')
			fc4 = slim.fully_connected(fc3, 10, activation_fn=tf.nn.softmax, scope='fc4')

			loss = slim.losses.softmax_cross_entropy(fc4, y)
			optimizer = tf.train.AdamOptimizer(learning_rate=config.AE_lr)

			return fc3, fc4, optimizer, loss 

	def run_model(self):
		#self.num_epoch = config.num_epoch
		#self.batch_size = config.batch_size
		batch_size=100
		num_epoch=1
		input_x = tf.placeholder(tf.float32, [batch_size,32,32,3], name='input_x')
		input_y = tf.placeholder(tf.float32, [batch_size,10], name='input_y')
		enc_fc3, output, optimizer, loss = self.model_architecture(input_x,input_y)

		with tf.Session as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			for current_epoch in range(num_epoch):
				idxs = np.random.permutation(self.train_x.shape[0])
				x = self.train_x[idxs]
				y = self.train_y[idxs]
				for i in range(batch_size):
					batch_x = x[i*batch_size:(i+1)*batch_size]
					batch_y = y[i*batch_size:(i+1)*batch_size]
					_, _loss_ = sess.run([optimizer, loss], feed_dict={input_x: batch_x, input_y: batch_y})
		    	if (current_epoch % 100 == 0):
					print('Epoch {}, Loss {}'.format(current_epoch+1, sess.run(loss, feed_dict={input_x: batch_x, input_y: batch_y})))

			saver.save(sess, './assets/AE-model.ckpt')

#run AE
model = Autoencoder()
model.run_model()
