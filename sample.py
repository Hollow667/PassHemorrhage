# Libraries
import os
import time
import pickle
import argparse
import tensorflow as tf
import numpy as np

# Files
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import utils
import models

input_dir = 'Generated_Password'
checkpoint = input("Enter the path to the pretained checkpoint : ")
output = input("Enter the name of the output file : ")
num_pass = int(input("Enter the number of passwords to be generated : "))
batch_size = 64
seq_length = 10
layer_dim = 128

if not os.path.isdir(input_dir):
    print('"{}" folder doesn\'t exist'.format(input_dir))

if not os.path.exists(checkpoint + '.meta'):
    print('"{}.meta" file doesn\'t exist'.format(checkpoint))

if not os.path.exists(os.path.join(input_dir, 'charmap.pickle')):
    print('charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(input_dir))

if not os.path.exists(os.path.join(input_dir, 'charmap_inv.pickle')):
    print('charmap_inv.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(input_dir))

# Dictionary
with open(os.path.join(input_dir, 'charmap.pickle'), 'rb') as f:
    charmap = pickle.load(f, encoding='latin1')

# Reverse-Dictionary
with open(os.path.join(input_dir, 'charmap_inv.pickle'), 'rb') as f:
    inv_charmap = pickle.load(f, encoding='latin1')


fake_inputs = models.Generator(batch_size, seq_length, layer_dim, len(charmap))

with tf.compat.v1.Session() as session:

    def generate_pass():
        pswrd = session.run(fake_inputs)
        pswrd = np.argmax(pswrd, axis=2)
        decoded_pswrd = []
        for i in range(len(pswrd)):
            decoded = []
            for j in range(len(pswrd[i])):
                decoded.append(inv_charmap[pswrd[i][j]])
            decoded_pswrd.append(tuple(decoded))
        return decoded_pswrd

    def save(pswrd):
        with open(output, 'a') as f:
                for p in pswrd:
                    p = "".join(s).replace('`', '')
                    f.write(p + "\n")

    saver = tf.compat.v1.train.Saver()
    saver.restore(session, checkpoint)

    passwords = []
    then = time.time()
    start = time.time()
    for i in range(int(num_pass / batch_size)):

        passwords.extend(generate_pass())

        # append to output file every 1000 batches
        if i % 1000 == 0 and i > 0:

            save(passwords)
            passwords = [] # flush

            print('wrote {} passwords to {} in {:.2f} seconds. {} total.'.format(1000 * batch_size, output, time.time() - then, i * batch_size))
            then = time.time()

    save(passwords)
print('\nFinished in {:.2f} seconds'.format(time.time() - start))
