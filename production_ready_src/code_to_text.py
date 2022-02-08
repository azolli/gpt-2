#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf

#relative imports
from encoder import get_encoder
from model import default_hparams
from sample import sample_sequence

class CodeToText: 
    def __init__(self, model_name="124M", seed=None, nsamples=1, batch_size=1, length=None, temperature=1, top_k=0, top_p=1, models_dir="models"):
        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.models_dir = models_dir
      
    def inference(self, t_checkpoint):

      models_dir = os.path.expanduser(os.path.expandvars(self.models_dir))
      if self.batch_size is None:
        self.batch_size = 1
      assert self.nsamples % self.batch_size == 0

      enc = get_encoder(self.model_name, models_dir)
      hparams = default_hparams()
      with open(os.path.join(models_dir, self.model_name, 'hparams.json')) as f:
          hparams.override_from_dict(json.load(f))

      if self.length is None:
        length = hparams.n_ctx // 2
      elif self.length > hparams.n_ctx:
          raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

      with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [self.batch_size, None])
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        output = sample_sequence(
            hparams=hparams, length=self.length,
            context=context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, self.model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = t_checkpoint
            while not raw_text:
                print('Prompt should not be empty!')
                break
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(self.nsamples // self.batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(self.batch_size)]
                })[:, len(context_tokens):]
                for i in range(self.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                return text

# provider = CodeToText(model_name="345M_finetuned", models_dir="/content/models", temperature=0.7, length=40)
# print(provider.inference("Defines a header 2"))

    

        