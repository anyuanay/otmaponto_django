import argparse
import ast
import json
import multiprocessing
import os
import time
import warnings
import pandas as pd

from gensim.models import fasttext  # version 3.8.3
from gensim.models.callbacks import CallbackAny2Vec

import nltk
from corpus_build_utils import finalize_corpus

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONHASHSEED"] = "17"


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""
    def __init__(self, epochs):
        self.epoch = 1
        self.epochs = epochs
        self.time = time.time()

    def on_epoch_begin(self, model):
        if self.epoch == 1:
            print("Starting computing of {} epochs".format(self.epochs))
        print(".", end="")

    def on_epoch_end(self, model):
        end = time.time()
        if self.epoch % 1 == 0 or self.epoch == self.epochs:
            print("Completed epoch {} of {}".format(self.epoch, self.epochs))
            print("Elapsed time was {t}".format(t=end - self.time))
            # print("Current loss is {}".format(model.get_latest_training_loss()))
        self.epoch += 1
        self.time = end


def train_ft_custom_model(_raw, _path, _model_name, _min_n, _max_n,
                          _window, _min_count, _epochs, _neg_samples):
    """
    A function to train a FastText model.
    :param _raw: The raw documents as a list of lists.
    :param _path: The name of the model to be saved.
    :param _model_name: The short name of the model.
    :param _min_n: The minimum n-gram characters.
    :param _max_n: The maximum n-gram characters.
    :param _window: The sliding window size.
    :param _min_count: Minimum number of times a word must appear.
    :param _epochs: Number of training epochs.
    :param _neg_samples: Number of negative samples.
    :return: A FastText model in binary form.
    """
    model_meta = {}
    model_params = {'min_n': _min_n,
                    'max_n': _max_n,
                    'window': _window,
                    'neg_samples': _neg_samples,
                    'min_count': _min_count,
                    'epochs': _epochs}
    epoch_logger = EpochLogger(_epochs)
    model = fasttext.FastText(sentences=_raw,
                              vector_size=300,
                              seed=17,
                              batch_words=5000,
                              epochs=model_params['epochs'],
                              sg=1,
                              ns_exponent=0.5,
                              sample=1e-5,
                              negative=model_params['neg_samples'],
                              window=model_params['window'],
                              min_n=model_params['min_n'],
                              max_n=model_params['max_n'],
                              min_count=model_params['min_count'],
                              # max_vocab_size=5200,
                              workers=multiprocessing.cpu_count(),
                              callbacks=[epoch_logger])
    # workers=multiprocessing.cpu_count() for training speedup at the cost of reproducibility
    print('Corpus has {n} documents'.format(n=len(_raw)))
    # model.build_vocab(_raw)
    # model.train(_raw,
    #            total_examples=model.corpus_count,
    #            epochs=model.iter)
    model.save(_path)
    model_meta['model_name'] = _model_name
    model_meta['parameters'] = model_params
    return model, model_meta

