import copy
import numpy as np
import os
import sys

import tensorflow as tf
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import applications, layers

from .callbacks import save_predictions, save_best_model

import FaceImageGenerator as FID

from reverse_gradient import reverse_gradient

DATA_DIR = '/vol/atlas/homes/dlt10/data/similarity_256_256'

# list = [datasets, batch, padding]
DATASETS = [
    ('disfa', 10, -1),
    # ('pain', 10, -1),
    ('fera2015', 10, -1),
    # ('imdb_wiki', 10, -1),
]

class AdversarialExpression(object):

    def __init__(self, args, datasets=DATASETS):
        self.args = args
        self.pip = FID.image_pipeline.FACE_pipeline(
            histogram_normalization=self.args.normalization,
            rotation_range=self.args.rotate,
            width_shift_range=self.args.transform,
            height_shift_range=self.args.transform,
            gaussian_range=self.args.gaussian_range,
            zoom_range=self.args.zoom,
            random_flip=True,
            allignment_type='similarity',
            grayscale=False,
            output_size=[256, 256],
            face_size=[224])
        self.datasets = datasets
        self.prediction_loss = []
        self.domain_loss = []
        self._load_data()

    def _load_data(self):
        load = FID.provider.flow_from_hdf5

        if self.args.trainingData == 'all':
            tr = [[load(os.path.join(DATA_DIR, '{}_{}.h5'.format(d, s)), b, p)
                   for s in ["tr", "te"]] for d, b, p in self.datasets]
        elif self.args.trainingData == 'tr':
            tr = [[load(os.path.join(DATA_DIR, '{}_{}.h5'.format(d, s)), b, p)
                   for s in ["tr"]] for d, b, p in self.datasets]
            te = [[load(os.path.join(DATA_DIR, '{}_{}.h5'.format(d, s)), b, p)
                   for s in ["te"]] for d, b, p in self.datasets]
        else:
            raise

        self.gen_tr_a = AdversarialExpression.generate_data(tr, True)
        self.gen_te_na = AdversarialExpression.generate_data(te, False)
        self.gen_tr_na = AdversarialExpression.generate_data(tr, False)

    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._load_resnet50()
            _, Y = next(self.gen_tr_a)
            with tf.variable_scope('label_predictor'):
                self._predict_labels(Y)
            with tf.variable_scope('domain_predictor'):
                self._predict_domains(Y)
            self._set_optimizer()

    def _load_resnet50(self):
        base_net = applications.resnet50.ResNet50(weights='imagenet')
        self.inp_0 = base_net.input
        base_net.layers.pop()
        base_net.outputs = [base_net.layers[-1].output]
        base_net.layers[-1].outbound_nodes = []
        self.Z = base_net.get_layer('flatten_1').output

    def _predict_labels(self, Y):
        for i, y in enumerate(Y[:-1]):
            name = self.datasets[i][0]
            logits = layers.Dense(
                y.shape[1], name="Dense_{}".format(name))(self.Z)
            skip = tf.not_equal(y, -1)
            skip = tf.reduce_min(tf.to_int32(skip), 1)
            skip = tf.to_float(skip)
            pred = tf.nn.softmax(logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y)
            pred_loss = skip * tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss_{}'.format(name), pred_loss)
            self.prediction_loss.append(pred_loss)

    def _predict_domains(self, Y):
        # Flip the gradient when backpropagating through this operation
        self.l = tf.placeholder(tf.float32, [])
        feat = reverse_gradient(self.Z, self.l)
        dp_fc0 = layers.Dense(100, activation='relu', name='dense_domain_relu',
                              kernel_initializer='glorot_normal')(feat)
        logits = layers.Dense(len(Y) - 1, activation='linear',
                              kernel_initializer='glorot_normal',
                              name='dense_domain_logit')(dp_fc0)
        pred = tf.nn.softmax(logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y[-1])
        domain_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('domain_loss', domain_loss)
        self.domain_loss.append(domain_loss)

    def _set_optimizer(self):
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.regular_train_op = tf.train.MomentumOptimizer(
            self.learning_rate, 0.9).minimize(tf.add_n(self.prediction_loss))
        total_loss = tf.add_n(self.prediction_loss + self.domain_loss)
        tf.summary.scalar('loss', total_loss)
        self.dann_train_op = tf.train.MomentumOptimizer(
            self.learning_rate, 0.9).minimize(total_loss)


    @staticmethod
    def generate_data(list_dat, aug, pip):
        out = []
        for i, dat in enumerate(list_dat):
            gen = dat[0]
            lab = next(gen['lab'])
            if lab.ndim == 3:
                lab = lab.argmax(2)
            lab = np.int8(lab)
            out.append(-np.ones_like(lab))

        while True:
            for i, dat in enumerate(list_dat):
                for gen in dat:
                    lab_list = copy.copy(out)

                    img = next(gen['img'])
                    lab = next(gen['lab'])

                    if lab.ndim == 3:
                        lab = lab.argmax(2)

                    lab_list[i] = lab

                    img_pp, _, _ =  pip.batch_transform(
                        img, preprocessing=True, augmentation=aug)

                    # crop the center of the image to size [244 244]
                    img_pp = img_pp[:, 16:-16, 16:-16, :]

                    img_pp -= np.apply_over_axes(np.mean, img_pp, [1, 2])
                    # add labels for domain classification
                    lab_list.append(np.ones_like(lab) * i)

                    yield [img_pp], lab_list

def train_model(args):

    pass
    # return model, pip