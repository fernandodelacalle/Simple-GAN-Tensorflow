import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import customs_ops as op
import bachdispenser as bachdispenser

def show_images_tensor(X):  
    numsubplots = np.ceil( np.sqrt( X.shape[0] ) ).astype(np.int8)
    # 8x8 only for represenatation
    fig, axes = plt.subplots(numsubplots, numsubplots, figsize=(8, 8))  
    for i, ax in enumerate(axes.flat):
        ax.imshow( 
            X[i,:,:],cmap = 'Greys_r',
            interpolation='nearest'
            )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def show_images_tensor_save(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        #plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        plt.imshow(sample.reshape(28, 28))
    return fig


# Simple generator and discriminator versions
# def generator_simple(input_noise, reuse = False, is_training=False):
#     u_1 = cop.linear_layer(input_noise, 'g_1', 128, reuse = reuse, is_training=is_training)
#     h_1 = tf.nn.relu(u_1)
#     u_2 = cop.linear_layer(h_1, 'g_2', 784, reuse = reuse, is_training=is_training)
#     h_2 = tf.nn.sigmoid(u_2)
#     h_2 = tf.reshape(h_2,  tf.pack([ tf.shape(h_2)[0],  28, 28, 1 ]) )   
#     return h_2

# def discriminator_simple(input_images, reuse = False, is_training=False):
#     x = tf.reshape(input_images,  tf.pack([ tf.shape(input_images)[0],  28*28 ]) )   
#     u_1 = cop.linear_layer(x, 'd_1', 128, reuse = reuse, is_training=is_training)
#     h_1 = tf.nn.relu(u_1)
#     logits = cop.linear_layer(h_1, 'd_2', 2, reuse = reuse, is_training=is_training)
#     return logits 


### working versions
# def generator(input_noise, reuse = False, is_training=False):

#     u_0 = op.linear_layer(input_noise, 'g_0', 100*2*7*7, reuse = reuse, is_training=is_training)
#     h_0 = tf.nn.relu(u_0)
#     h_0 = tf.reshape(h_0,  tf.pack( [ tf.shape(h_0)[0], 7, 7, 100 * 2] )   ) 
    
#     u_1 = op.deconv2d(h_0, 'g_1', [ tf.shape(h_0)[0], 7*2, 7*2, 100 * 1]  , reuse = reuse) 
#     h_1 = tf.nn.relu(u_1)

#     u_2 = op.deconv2d(h_1, 'g_2', [ tf.shape(h_1)[0], 7*2*2, 7*2*2, 1], reuse = reuse) 
#     h_2 = tf.nn.sigmoid(u_2)
    
#     return h_2

# def discriminator(input_images, reuse = False, is_training=False):
#     x = tf.reshape(input_images,  tf.pack([ tf.shape(input_images)[0],  28*28 ]) )   
    
#     u_1 = op.linear_layer(x, 'd_1', 128, reuse = reuse, is_training=is_training)
    
#     h_1 = tf.nn.relu(u_1)
    
#     logits = op.linear_layer(h_1, 'd_2', 2, reuse = reuse, is_training=is_training)
    
#     return logits 

## Fully convolutional generator and discriminator
def generator(input_noise, reuse = False, is_training=False):

    u_0 = op.linear_layer(input_noise, 'g_0', 100*2*7*7, reuse = reuse, is_training=is_training)
    # u_0 = tf.contrib.layers.batch_norm(u_0, center=True, scale=True, 
    #                                       is_training=is_training, reuse = reuse,
    #                                       scope='g_0_bn')
    h_0 = tf.nn.relu(u_0)
    h_0 = tf.reshape(h_0,  tf.pack( [ tf.shape(h_0)[0], 7, 7, 100 * 2] )   ) 
    

    u_1 = op.deconv2d(h_0, 'g_1', [ tf.shape(h_0)[0], 7*2, 7*2, 100 * 1]  , reuse = reuse) 
    # u_1 = tf.contrib.layers.batch_norm(u_1, center=True, scale=True, 
    #                                       is_training=is_training, reuse = reuse,
    #                                       scope='g_1_bn')
    h_1 = tf.nn.relu(u_1)


    u_2 = op.deconv2d(h_1, 'g_2', [ tf.shape(h_1)[0], 7*2*2, 7*2*2, 1], reuse = reuse) 
    h_2 = tf.nn.sigmoid(u_2)
    
    return h_2




def discriminator(input_images, reuse = False, is_training=False):
    #x = tf.reshape(input_images,  tf.pack([ tf.shape(input_images)[0],  28*28 ]) )   
    
    d = 32

    u_0 = op.convolution(input_images, 'd_0', 1, d, 3, reuse=reuse, is_training=is_training, stride = 2)
    h_0 = op.lrelu(u_0)
    # if is_training:
    #     h_0 = tf.nn.dropout(h_0, 0.25)


    u_1 = op.convolution(h_0, 'd_1', d, d*2, 3, reuse=reuse, is_training=is_training, stride = 2)
    h_1 = op.lrelu(u_1)
    # if is_training:
    #     h_1 = tf.nn.dropout(h_1, 0.25)

    dim = h_1.get_shape()
    h_1 = tf.reshape(h_1,  tf.pack( [ tf.shape(h_1)[0], dim[1] * dim[2] * dim[3]] )   ) 

    logits = op.linear_layer(h_1, 'd_2', 2, reuse = reuse, is_training=is_training)
    return logits 


input_noise = tf.placeholder("float", [None, 100] )
input_images = tf.placeholder("float", [None, 28, 28, 1])
y_ = tf.placeholder("float", [None, 2])

gen = generator(input_noise, reuse = False, is_training=False )
dis_logits = discriminator(input_images, reuse = False, is_training=True)
gan_logits = discriminator(generator(input_noise, reuse = True, is_training=True), reuse = True, is_training=False)

d_cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=dis_logits))
g_cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=gan_logits))


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
d_train_step = tf.train.AdamOptimizer().minimize(d_cross_entropy, var_list=d_vars)
g_train_step = tf.train.AdamOptimizer().minimize(g_cross_entropy, var_list=g_vars)


bd = bachdispenser.BatchDispenser()
bd.load_dataset()

bach_size = 64

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

    noise = np.random.uniform(0,1,size=[16 ,100])
    x_gen = sess.run(gen, feed_dict={input_noise: noise    })
    
    fig = show_images_tensor_save(x_gen[0:16,:,:,0])
    plt.savefig('out/{}.png'.format( 'initial'), bbox_inches='tight')
    plt.close(fig)

    for step in xrange(100000):
        g_noise = np.random.uniform(-1,1,size=[bach_size,100])
        x_gen = sess.run(gen, feed_dict={input_noise: g_noise    })
        
        x_real, y = bd.next_batch_matrix(bach_size)
        x_real = np.expand_dims(x_real, 3)  
        x_batch = np.append(x_real, x_gen,0)

        labels = np.zeros( (bach_size*2,2) )
        labels[:bach_size, 0] = 1
        labels[bach_size:, 1] = 1
        
        _, d_loss  = sess.run([d_train_step, d_cross_entropy], feed_dict={input_images: x_batch,  y_ : labels    }    )

        noise = np.random.uniform(-1,1,size=[bach_size,100])
        labels = np.zeros( (bach_size,2) )
        labels[:, 0] = 1
        
        _, g_loss = sess.run( [g_train_step, g_cross_entropy],   feed_dict={input_noise: noise,  y_ : labels    }    )

        if step % 1000 == 0:
            print g_loss
            print d_loss 
            fig = show_images_tensor_save(x_gen[0:16,:,:,0])
            plt.savefig('out/{}.png'.format(str(step)), bbox_inches='tight')
            plt.close(fig)

import pickle
with open('final.pickle', 'w') as f: 
    pickle.dump(x_gen, f)


