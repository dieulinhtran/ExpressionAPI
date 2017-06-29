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


def mse(y_true, y_pred):
    '''
    '''
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip), 1)
    skip = tf.to_float(skip)
    cost = tf.reduce_mean(tf.square(y_true-y_pred), 1)
    return cost * skip

def domain(y_true, y_pred):
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip), 1)
    skip = tf.to_float(skip)
    cost = tf.reduce_mean(tf.square(y_true-y_pred), 1)


def train_model(args):

    pip = FID.image_pipeline.FACE_pipeline(
        histogram_normalization=args.normalization,
        rotation_range=args.rotate,
        width_shift_range=args.transform,
        height_shift_range=args.transform,
        gaussian_range=args.gaussian_range,
        zoom_range=args.zoom,
        random_flip=True,
        allignment_type='similarity',
        grayscale=False,
        output_size=[256, 256],
        face_size=[224])

    def data_provider(list_dat, aug):

        out = []
        for dat in list_dat:
            gen = dat[0]
            img = next(gen['img'])

            lab = next(gen['lab'])
            if lab.ndim == 3:
                lab = lab.argmax(2)

            lab = np.int8(lab)
            out.append(-np.ones_like(lab))

        while True:
            n_domains = len(list_dat)
            for i, dat in enumerate(list_dat):
                for gen in dat:
                    lab_list = copy.copy(out)

                    img = next(gen['img'])
                    lab = next(gen['lab'])

                    if lab.ndim == 3:
                        lab = lab.argmax(2)

                    lab_list[i] = lab

                    img_pp, _, _ = pip.batch_transform(
                        img, preprocessing=True, augmentation=aug)

                    # crop the center of the image to size [244 244]
                    img_pp = img_pp[:, 16:-16, 16:-16, :]

                    img_pp -= np.apply_over_axes(np.mean, img_pp, [1, 2])

                    yield [img_pp], lab_list

    datasets_batch_padding = [
        ('disfa', 10, -1),
        # ('pain', 10, -1),
        ('fera2015', 10, -1),
        # ('imdb_wiki', 10, -1),
    ]

    load = FID.provider.flow_from_hdf5

    if args.trainingData == 'all':
        TR = [[load(os.path.join(DATA_DIR, '{}_{}.h5'.format(d, s)), b, p)
               for s in ["tr", "te"]] for d, b, p in datasets_batch_padding]
        pass
    elif args.trainingData == 'tr':
        TR = [[load(os.path.join(DATA_DIR, '{}_{}.h5'.format(d, s)), b, p)
               for s in ["tr"]] for d, b, p in datasets_batch_padding]
        TE = [[load(os.path.join(DATA_DIR, '{}_{}.h5'.format(d, s)), b, p)
               for s in ["te"]] for d, b, p in datasets_batch_padding]
    else:
        raise

    GEN_TR_a = data_provider(TR, True)
    GEN_TE_na = data_provider(TE, False)
    GEN_TR_na = data_provider(TR, False)

    X, Y = next(GEN_TR_a)
    print("X", len(X))
    print("Y", len(Y))

    base_net = applications.resnet50.ResNet50(weights='imagenet')
    base_net.summary()

    inp_0 = base_net.input
    base_net.layers.pop()
    base_net.outputs = [base_net.layers[-1].output]
    base_net.layers[-1].outbound_nodes = []
    Z = base_net.get_layer('flatten_1').output
    # Z = base_net.get_layer('fc1000').output
    out, loss = [], []
    for i, y in enumerate(Y):
        print("y shape", y.shape)
        net = layers.Dense(y.shape[1], name="Dense_{}".format(
            datasets_batch_padding[i][0]))(Z)
        out.append(net)
        loss.append(mse)
    #out.append()
    print("i", i)
    # sys.exit()
    with tf.variable_scope('domain_predictor'):
                    
        # Flip the gradient when backpropagating through this operation
        feat = reverse_gradient(Z, self.l)
        dp_fc0 = layers.Dense(100, activation='relu', name='dense_domain_1'
                              kernel_initializer='glorot_normal')(feat)
        dp_fc1 = layers.Dense(len(Y) - 1, activation='softmax',
                              kernel_initializer='glorot_normal',
                              name='dense_domain_2')(dp_fc0)
        domain_loss = K.categorical_crossentropy(dp_fc1, Y[-1])
        out.append(dp_fc1)
        loss.append(domain_loss)

    model = K.models.Model([inp_0], out)
    model.summary()

    model.compile(
        optimizer=K.optimizers.Adadelta(
            lr=1.,
            rho=0.95,
            epsilon=1e-08,
            decay=1e-5,
        ),
        loss=loss,
        metrics=[mse]
    )

    model.fit_generator(
        generator=GEN_TR_a,
        steps_per_epoch=2000,
        epochs=args.epochs,
        max_q_size=100,
        validation_data=GEN_TE_na,
        validation_steps=100,
        callbacks=[
            save_predictions(GEN_TR_na, args.log_dir+'/TR_'),
            save_predictions(GEN_TE_na, args.log_dir+'/TE_'),
            save_best_model(GEN_TE_na, args.log_dir),
            K.callbacks.ModelCheckpoint(
                args.log_dir + '/model.h5', save_weights_only=True)
            ,
            K.callbacks.TensorBoard(log_dir=args.log_dir + '/Graph',
                                    histogram_freq=0, write_graph=True,
                                    write_images=True)]
        )

    return model, pip
