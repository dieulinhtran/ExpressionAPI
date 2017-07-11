import copy
import h5py
from itertools import islice
from math import floor
import numpy as np
import os
import sys
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import applications, layers

from .callbacks import save_predictions, save_best_model

import FaceImageGenerator as FID

from .reverse_gradient import GradientReversal

DATA_DIR = '/vol/atlas/homes/dlt10/data/similarity_256_256'

# list = [datasets, batch, padding]
DATASETS = [
    ('disfa', 10, -1),
    # ('pain', 10, -1),
    ('fera2015', 10, -1),
    # ('imdb_wiki', 10, -1),
]


class FacialExpressionDatasets(object):
    def __init__(self, args, datasets=DATASETS, one_hot=False):
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
        self.one_hot = one_hot
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

        # train + augmentation 
        self.gen_tr_a = FacialExpressionDatasets.gen_data(
            tr, True, self.pip, one_hot=self.one_hot)
        # count number of examples in train 
        self._n_train_examples = 0
        for dataset_name, _, _ in self.datasets:
            file_name = '{}_{}.h5'.format(dataset_name, "tr")
            with h5py.File(os.path.join(DATA_DIR, file_name)) as f:
                self._n_train_examples += len(f['lab'])
        # train + no augmentation 
        self.gen_tr_na = FacialExpressionDatasets.gen_data(
            tr, False, self.pip, one_hot=self.one_hot)
        self._n_test_examples = 0
        if self.args.trainingData == 'tr':
            # count number of examples in train 
            self.gen_te_na = FacialExpressionDatasets.gen_data(
                te, False, self.pip, one_hot=self.one_hot)
            for dataset_name, _, _ in self.datasets:
                file_name = '{}_{}.h5'.format(dataset_name, "te")
                with h5py.File(os.path.join(DATA_DIR, file_name)) as f:
                    self._n_test_examples += len(f['lab'])


    @staticmethod
    def gen_data(list_dat, aug, pip, one_hot=False):
        out = []
        domains = {}
        for i, dat in enumerate(list_dat):
            gen = dat[0]
            lab = next(gen['lab'])
            if lab.ndim == 3 and not one_hot:
                lab = lab.argmax(2)
            lab = np.int8(lab)
            out.append(-np.ones_like(lab))
            domains[lab.shape[1]] = i
        n_domains = len(domains.keys())
        n_samples = lab.shape[0]

        while True:
            for i, dat in enumerate(list_dat):
                for gen in dat:
                    lab_list = copy.copy(out)

                    img = next(gen['img'])
                    lab = next(gen['lab'])

                    if lab.ndim == 3 and not one_hot:
                        lab = lab.argmax(2)

                    lab_list[i] = lab
                    domain_lab = np.zeros((n_samples, n_domains))
                    domain_lab[:, domains[lab.shape[1]]] = 1
                    lab_list.append(domain_lab)

                    img_pp, _, _ =  pip.batch_transform(
                        img, preprocessing=True, augmentation=aug)

                    # crop the center of the image to size [244 244]
                    img_pp = img_pp[:, 16:-16, 16:-16, :]

                    img_pp -= np.apply_over_axes(np.mean, img_pp, [1, 2])
                    # add labels for domain classification

                    yield [img_pp], lab_list

    @property
    def n_train_examples(self):
        return self._n_train_examples

    @property
    def n_test_examples(self):
        return self._n_test_examples

    def load_test(self):
        pass
        



class AdversarialExpression(object):

    def __init__(self, args, Y, datasets=DATASETS, adversarial=True):
        self.args = args
        self.adversarial = adversarial
        self.datasets = datasets
        self.prediction_loss = []
        self.domain_loss = []
        self.prediction_logits = []
        self.prediction_skips = []
        self.y = []
        self._build_graph(Y)


    def _build_graph(self, Y):
        self._load_resnet50()
        self.session = K.get_session() 
        self.graph = self.session.graph
        with tf.variable_scope('label_predictor'):
            self._predict_labels(Y[:-1])
        if self.adversarial:
            with tf.variable_scope('domain_predictor'):
                self._predict_domains(Y)
        self._set_optimizer()

    def _load_resnet50(self):
        base_net = applications.resnet50.ResNet50(weights='imagenet')
        self.x_in = base_net.input
        base_net.layers.pop()
        base_net.outputs = [base_net.layers[-1].output]
        base_net.layers[-1].outbound_nodes = []
        self.Z = base_net.get_layer('flatten_1').output

    def _predict_labels(self, Y):
        for i, y in enumerate(Y):
            # create placeholder according to output shape
            _, n_classes, n_outputs = y.shape
            y_ = tf.placeholder(tf.float32, [None, n_classes, n_outputs])
            print(y_)
            self.y.append(y_)
            name = self.datasets[i][0]
            # skip = 0 if the labels are all set to -1
            skip = tf.not_equal(y, -1)
            skip = tf.reduce_min(tf.to_int32(skip), 1)
            skip = tf.to_float(skip)
            skip = tf.Print(skip, [skip], "skip")
            self.prediction_skips.append(skip)
            # creating logits 
            dense_name = "Dense_{}".format(name)
            logits = layers.Dense(n_classes, name=dense_name)(self.Z)
            self.prediction_logits.append(logits)
            # use cross entropy for loss function 
            # condition on skip > 0
            def mean_cross_entropy(): 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=y_)
                return tf.reduce_mean(cross_entropy) 
            pred_loss = tf.cond(skip == 0., skip, mean_cross_entropy)
            pred_loss = skip * tf.reduce_mean(cross_entropy)
            pred_loss = tf.Print(pred_loss, [pred_loss], "pred_loss_{}".format(i))
            self.prediction_loss.append(pred_loss)
            # tensorboard
            tf.summary.scalar('loss_{}'.format(name), pred_loss)

    def _predict_domains(self, Y):
        # Flip the gradient when backpropagating through this operation
        n_domain = len(Y) - 1
        self.l = tf.placeholder(tf.float32, [])
        self.domain_labels = tf.placeholder(tf.float32, [None, n_domains])
        feat = GradientReversal()(self.Z, self.l)
        dp_fc0 = layers.Dense(100, activation='relu', name='dense_domain_relu',
                              kernel_initializer='glorot_normal')(feat)
        self.domain_logits = layers.Dense(
            n_domains, activation='linear', name='dense_domain_logit',
            kernel_initializer='glorot_normal')(dp_fc0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.domain_logits, labels=self)
        domain_loss = tf.reduce_mean(cross_entropy)
        self.domain_loss.append(domain_loss)
        # tensorboard
        tf.summary.scalar('domain_loss', domain_loss)

    def _set_optimizer(self):
        self.learning_rate = tf.placeholder(tf.float32, [])
        if self.adversarial:
            self.total_loss = tf.add_n(self.prediction_loss + self.domain_loss)
            self.optimizer = tf.train.MomentumOptimizer(
                self.learning_rate, 0.9).minimize(self.total_loss)
        else:
            self.total_loss = tf.add_n(self.prediction_loss)
            self.optimizer = tf.train.MomentumOptimizer(
                self.learning_rate, 0.9).minimize(self.total_loss)
        # tensorboard
        tf.summary.scalar('loss', self.total_loss)


def train_adversarial(args):
    # get data
    # list = [datasets, batch, padding]
    DATA = [
        ('disfa', args.batch_size, -1),
        # ('pain', 10, -1),
        ('fera2015', args.batch_size, -1),
        # ('imdb_wiki', 10, -1),
    ]
    face_datasets = FacialExpressionDatasets(
        args, datasets=DATASETS, one_hot=True)
    gen_tr_na  = face_datasets.gen_tr_na
    gen_te_na  = face_datasets.gen_te_na
    X, Y = next(gen_tr_na)
    print(X[0].shape)
    print([y.shape for y in Y])

    adv_expr = AdversarialExpression(args, Y, datasets=DATA, adversarial=False)
    with adv_expr.session as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps_per_epoch = floor(face_datasets.n_train_examples/args.batch_size)
        preds = zip(adv_expr.prediction_logits,
                    adv_expr.prediction_skips,
                    adv_expr.y)
        acc_list = []
        for pred, skip, labels in preds:
            correct_prediction = tf.equal(
                tf.argmax(pred, 1), tf.argmax(labels, 1))
            acc = skip * tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            acc_list.append(acc)
        accuracy = tf.add_n(acc_list)

        for epoch in range(args.epochs):
            print("Epoch {}/{}".format(epoch + 1, args.epochs))
            avg_cost = 0.
            for i in tqdm(range(steps_per_epoch)):
                batch_x, batch_y = next(gen_tr_na)
                print([y.shape for y in batch_y])
                feed_dict = {
                    K.learning_phase(): 1,
                    adv_expr.x_in: batch_x[0],
                    adv_expr.learning_rate: 0.01,
                    # adv_expr.y[0]: batch_y[0],
                    # adv_expr.y[1]: batch_y[1],
                    # adv_expr.domain_labels: batch_y[2]
                }
                z = sess.run([adv_expr.Z], feed_dict=feed_dict) 
                print(z[0].shape)
                _, l = sess.run([adv_expr.optimizer, adv_expr.total_loss],
                                 feed_dict=feed_dict)
                avg_cost += l/steps_per_epoch
            print("avg_cost", avg_cost)
            '''
            print("Loss (train): {:.04f} \t Accuracy (test):  {:.04f}".format(
                avg_cost, accuracy.eval({adv_expr.x_in: test_x,
                                         labels: mnist.test.labels})))
            '''
