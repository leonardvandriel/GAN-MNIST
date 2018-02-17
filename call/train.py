import os
import numpy as np
from model import *
from util import *
from load import call_labels, call_set
import random

labels = call_labels()
iters = 4000
learning_rate = 0.0002
batch_size = 128
image_shape = [112, 112, 3]
dim_z = 100
dim_y = len(labels)
dim_W1 = 1024
dim_W2 = 256
dim_W3 = 128
dim_W4 = 64
dim_channel = image_shape[2]

visualize_dim = 196

label_t, image_t, onehots = call_set(labels, image_shape, batch_size)

dcgan_model = DCGAN(
    batch_size=batch_size,
    image_shape=image_shape,
    dim_z=dim_z,
    dim_y=dim_y,
    dim_W1=dim_W1,
    dim_W2=dim_W2,
    dim_W3=dim_W3,
    dim_W4=dim_W4,
    dim_channel=dim_channel,
)

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model(
)
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'),
                      tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = tf.train.AdamOptimizer(
    learning_rate, beta1=0.5).minimize(
        d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(
    learning_rate, beta1=0.5).minimize(
        g_cost_tf, var_list=gen_vars)

Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(
    batch_size=visualize_dim)

tf.global_variables_initializer().run()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim, dim_z))
Y_np_sample = OneHot(np.random.randint(dim_y, size=[visualize_dim]))

step = 20
d_loss_max = 1.3
p_real_min = 0.5
randomness = 0.1

p_real_last, p_gen_last, g_loss_last, d_loss_last = 0.5, 0.5, 1, 1
g_updates = 0

print('d_loss_max', d_loss_max)
print('p_real_min', p_real_min)
print('randomness', randomness)

if True:
    for iterations in range(iters + 1):

        images, labels = sess.run([image_t, label_t])
        Xs = images / 255.
        Ys = np.array(
            [onehots[f.decode('utf-8').split('/')[2]] for f in labels])
        Zs = np.random.uniform(
            -1, 1, size=[batch_size, dim_z]).astype(np.float32)

        update_g = (random.random() > randomness) == (d_loss_last < d_loss_max and p_real_last > p_real_min)
        if update_g:
            g_updates += 1
            _, gen_loss_val = sess.run(
                [train_op_gen, g_cost_tf], feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys
                })
            discrim_loss_val, p_real_val, p_gen_val = sess.run(
                [d_cost_tf, p_real, p_gen],
                feed_dict={
                    Z_tf: Zs,
                    image_tf: Xs,
                    Y_tf: Ys
                })

        else:
            _, discrim_loss_val = sess.run(
                [train_op_discrim, d_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys,
                    image_tf: Xs
                })
            gen_loss_val, p_real_val, p_gen_val = sess.run(
                [g_cost_tf, p_real, p_gen],
                feed_dict={
                    Z_tf: Zs,
                    image_tf: Xs,
                    Y_tf: Ys
                })

        g_loss_last = gen_loss_val
        d_loss_last = discrim_loss_val
        p_real_last = p_real_val.mean()
        p_gen_last = p_gen_val.mean()
        print("u:%s i:%04d G/D:%.3f Lg:%.4f Ld:%.4f Pr:%.4f Pg:%.4f" % (("G" if update_g else "D"), iterations, 1.0 * g_updates / (iterations + 1), gen_loss_val, discrim_loss_val, p_real_last, p_gen_last))

        if np.mod(iterations, step) == 0:
            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: Z_np_sample,
                    Y_tf_sample: Y_np_sample
                })
            generated_samples = (generated_samples + 1.) / 2.
            if iterations == 0:
                save_visualization(images, (14,14), save_path='./vis/input_%04d.jpg' % int(iterations/step))
            save_visualization(
                generated_samples, (14, 14),
                save_path='./vis/sample_%04d.jpg' % int(iterations / step))

coord.request_stop()
coord.join(threads)
