import numpy as np
import tensorflow as tf

bias_init = 0.0

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear_layer(x, name_layer, hidden_units, reuse = False, is_training=False):
    with tf.variable_scope(name_layer, reuse=reuse):
        in_dim = int(x.get_shape()[1])
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        weights = tf.get_variable(
                    'weights', [x.get_shape()[1], hidden_units],
                    initializer=tf.random_normal_initializer(stddev=xavier_stddev))
        biases = tf.get_variable(
                    'biases', [hidden_units],
                    initializer=tf.constant_initializer(bias_init))
        x = tf.matmul(x, weights) + biases
    return x

def convolution(inputs_img, name_layer, in_dim, out_dim, conv_size, reuse=False, is_training=False, stride = 1):
    with tf.variable_scope(name_layer + '_parameters', reuse=reuse):
        n = conv_size*conv_size*out_dim
        weights = tf.get_variable('weights_'+name_layer, [conv_size, conv_size, in_dim, out_dim],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        biases = tf.get_variable('biases_'+name_layer,   [out_dim],   initializer=tf.constant_initializer(bias_init) )
    with tf.variable_scope(name_layer + 'conv', reuse=reuse):
        conv = tf.nn.conv2d(inputs_img,  weights, [1, stride, stride, 1], padding='SAME')
        #hidden = conv + biases
        hidden = conv 
    return hidden

def conv2d(input_, name, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, reuse = False):
    with tf.variable_scope(name, reuse=reuse):
        in_dim = k_h*k_w*output_dim
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=xavier_stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(bias_init))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)
        return conv

def deconv2d(input_, name, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_shape[-1]], initializer= tf.constant_initializer(bias_init))  
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=tf.pack(output_shape), strides=[1, d_h, d_w, 1])
    deconv = tf.nn.bias_add(deconv, biases)
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.pack(output_shape) )
    return deconv
    