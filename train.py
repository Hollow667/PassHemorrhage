import os, sys
sys.path.append(os.getcwd())

import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import models

from passgen import *

tf.compat.v1.disable_eager_execution()

def append_new_line(file , text):
    with open(file, 'a+') as f:
        f.seek(0)
        accepted_pass = f.read(100)
        if len(accepted_pass) > 0:
            f.write("\n")
        f.write(text)

main()
for words in generate_wordlist.unique_list_finished:
        append_new_line('dataset/text.txt', words)

data = 'dataset/text.txt'


output_dir = 'Generated_Password'
seq_length = 10
batch_size = 64
layer_dim = 128
critic_iters = 10
lamb = 10
iters = 2000
save_every = 500
training_data = data


lines, charmap, inv_charmap = utils.load_dataset(
    path=training_data,
    max_length=seq_length)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(os.path.join(output_dir, 'checkpoints')):
    os.makedirs(os.path.join(output_dir, 'checkpoints'))

if not os.path.isdir(os.path.join(output_dir, 'pass_generated')):
    os.makedirs(os.path.join(output_dir, 'pass_generated'))

with open(os.path.join(output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f)

with open(os.path.join(output_dir, 'charmap_inv.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f)

print("Number of unique characters in dataset: {}".format(len(charmap)))

real_inputs_discrete = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, seq_length])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

fake_inputs = models.Generator(batch_size, seq_length, layer_dim, len(charmap))
fake_inputs_discrete = tf.argmax(input=fake_inputs, axis=fake_inputs.get_shape().ndims-1)

disc_real = models.Discriminator(real_inputs, seq_length, layer_dim, len(charmap))
disc_fake = models.Discriminator(fake_inputs, seq_length, layer_dim, len(charmap))

disc_cost = tf.reduce_mean(input_tensor=disc_fake) - tf.reduce_mean(input_tensor=disc_real)
gen_cost = -tf.reduce_mean(input_tensor=disc_fake)

alpha = tf.random.uniform(
    shape=[batch_size,1,1],
    minval=0.,
    maxval=1.
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(ys=models.Discriminator(interpolates, seq_length, layer_dim, len(charmap)), xs=[interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients), axis=[1,2]))
gradient_penalty = tf.reduce_mean(input_tensor=(slopes-1.)**2)
disc_cost += lamb * gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-batch_size+1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+batch_size]],
                dtype='int32'
            )

true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[10*batch_size:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*batch_size], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]


with tf.compat.v1.Session() as session:

  localtime = time.asctime( time.localtime(time.time()) )
  print("Starting TensorFlow session...")
  print("Local current time :", localtime)

  session.run(tf.compat.v1.global_variables_initializer())

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

  gen = inf_train_gen()

  for iteration in range(iters + 1):
      start_time = time.time()

      if iteration > 0:
          _ = session.run(gen_train_op)

      for i in range(critic_iters):
          _data = next(gen)
          _disc_cost, _ = session.run(
              [disc_cost, disc_train_op],
              feed_dict={real_inputs_discrete:_data}
          )

      if iteration % 100 == 0 and iteration > 0:
        passwords = []
        for i in range(10):
            passwords.extend(generate_pass())

        for p in passwords:
            p = "".join(p).replace('`', '')
            append_new_line(os.path.join(output_dir, 'pass_generated', 'passwords_{}.txt').format(iteration), p)

      print(f"\nTraining {iteration}/{iters} ({iteration/iters*100})")
      for i in tqdm(range(100)):
          time.sleep(0.1)
          if iteration % save_every == 0 and iteration > 0:
              model_saver = tf.compat.v1.train.Saver()
              model_saver.save(session, os.path.join(output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))

      if iteration == iters:
          print("...Training done.")

localtime = time.asctime( time.localtime(time.time()) )
print("Ending TensorFlow session.")
print("Local current time :", localtime)
