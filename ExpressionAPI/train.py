import copy
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import applications, layers

from .callbacks import save_predictions, save_best_model

import FaceImageGenerator as FID

DATA_DIR = '/vol/atlas/homes/dlt10/data/similarity_256_256'


def mse(y_true, y_pred):
    '''
    '''
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip), 1)
    skip = tf.to_float(skip)
    cost = tf.reduce_mean(tf.square(y_true-y_pred), 1)
    return cost * skip


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
        ('fera', 10, -1),
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

    base_net = applications.resnet50.ResNet50(weights='imagenet')

    inp_0 = base_net.input
    Z = base_net.get_layer('flatten_1').output

    out, loss = [], []
    for i, y in enumerate(Y):
        net = layers.Dense(y.shape[1])(Z)
        out.append(net)
        loss.append(mse)

    model = K.models.Model([inp_0], out)
    model.summary()

    model.compile(
        optimizer=K.optimizers.Adadelta(
            lr=1.,
            rho=0.95,
            epsilon=1e-08,
            decay=1e-5,
        ),
        loss=loss
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
            ]
        )

    return model, pip
