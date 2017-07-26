import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from Autoencoder import model_architecture

def gen1(y, batch_size=config.batch_size):
	with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu): 
		z1 = tf.random_uniform([batch_size,50],0,1)
		gen1_1 = slim.batch_norm(slim.fully_connected(z1, 256), scope='gen1_1')
		gen1_2 = slim.batch_norm(slim.fully_connected(y, 512), scope='gen1_2')
		z = tf.concat([gen1_1, gen1_2], 1)
		z = slim.batch_norm(slim.fully_connected(z, 512), scope='gen1_3')
		z = slim.batch_norm(slim.fully_connected(z, 512), scope='gen1_4')
		z = slim.fully_connected(z, 256), scope='gen1_5')

		return z
        
def gen0(x, y, batch_size=config.batch_size)
	with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], activation_fn=tf.nn.relu):
		z0 = tf.random_uniform([batch_size,16],0,1)
		z_embed = slim.batch_norm(slim.fully_connected(slim.batch_norm(slim.fully_connected(z0, 128)), 128), scope='gen0_embed')
		with tf.Session as sess():
			saver.restore(sess, './assets/AE-model.ckpt')
			fc3, _1, _2, _3 = model_architecture(x, y)
			enc_fc3 = sess.run([fc3], feed_dict{x: x, y: y})
		gen0_input = tf.concat([enc_fc3, z_embed], 1)
		next_layer = batch_norm(slim.fully_connected(gen0_input, num_output1*5*5, activation_fn=tf.nn.relu))
		next_layer = reshape(next_layer, (batch_size,num_output1,5,5))
		next_layer = batch_norm(deconv2d(next_layer, (batch_size,num_output1,10,10),

