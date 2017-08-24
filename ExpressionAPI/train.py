import os
import argparse
import pickle
import copy
import numpy as np
import sys
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import applications, layers
from tensorflow.contrib.keras import backend as K

from .callbacks import save_predictions, save_best_model
from .gradient_reversal import GradientReversal
import FaceImageGenerator as FID

DATA_DIR = '/vol/hmi/projects/linh/data/similarity_256_256'


def mse(y_true, y_pred):
    '''
    '''
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip),1)
    skip = tf.to_float(skip)
    cost = tf.reduce_mean(tf.square(y_true-y_pred),1)
    return cost * skip


def Resnet50DomainAdaptation(X, Y, weights='imagenet', dataset_names=[]):
    base_net = applications.resnet50.ResNet50(weights=weights)
    inp_0 = base_net.input
    base_net.layers.pop()
    base_net.outputs = [base_net.layers[-1].output]
    base_net.layers[-1].outbound_nodes = []
    Z = base_net.get_layer('flatten_1').output
    out, loss = [], []

    # prediction loss
    for i, y in enumerate(Y[:-1]):
    # for i, y in enumerate(Y):
        if dataset_names:
            dataset_name = dataset_names[i]
        else:
            dataset_name = i
        net = layers.Dense(y.shape[1], name="dense_{}".format(
            dataset_name))(Z)
        out.append(net)

    # domain loss
    # Flip the gradient when backpropagating through this operation
    grl = GradientReversal(1.0)
    feat = grl(Z)
    dp_fc0 = layers.Dense(100, activation='relu', name='dense_domain_relu',
                          kernel_initializer='glorot_normal')(feat)
    domain_logits = layers.Dense(len(Y) - 1, activation='linear',
                          kernel_initializer='glorot_normal',
                          name='dense_domain_logit')(dp_fc0)
    out.append(domain_logits)

    # initialize model
    model = keras.models.Model(inp_0, out)
    return model


def data_provider(list_dat, aug, pip):

    out = []
    for dat in list_dat:
        gen = dat[0]
        img = next(gen['img'])

        lab = next(gen['lab'])
        if lab.ndim==3:
            lab = lab.argmax(2)

        lab = np.int8(lab)

        out.append(-np.ones_like(lab))
    n_domains = len(list_dat)


    while True:
        for i, dat  in enumerate(list_dat):
            for gen in dat:
                lab_list = copy.copy(out)

                img = next(gen['img'])
                lab = next(gen['lab'])

                if lab.ndim==3:lab = lab.argmax(2)

                lab_list[i] = lab

                img_pp, _, _  = pip.batch_transform(
                    img, preprocessing=True, augmentation=aug)

                # crop the center of the image to size [244 244]
                img_pp = img_pp[:,16:-16,16:-16,:]

                img_pp -= np.apply_over_axes(np.mean, img_pp, [1,2])
                # add labels for domain classification
                
                lab_domain = np.ones((lab.shape[0], n_domains))
                lab_domain[:, i] = 1.
                lab_list.append(lab_domain)
                
                yield img_pp, lab_list


def trange(*args, **kwargs):
    """A shortcut for writing tqdm(range)"""
    return tqdm(range(*args), **kwargs)


def train_model(args):

    pip = FID.image_pipeline.FACE_pipeline(
            histogram_normalization = args.normalization,
            rotation_range = args.rotate,
            width_shift_range = args.transform,
            height_shift_range = args.transform,
            gaussian_range = args.gaussian_range,
            zoom_range = args.zoom,
            random_flip = True,
            allignment_type = 'similarity',
            grayscale = False,
            output_size = [256, 256],
            face_size = [224],
            )

    datasets_batch_padding = [
        ('disfa', args.batch, -1),
        # ('pain', 10, -1),
        ('fera2015', args.batch, -1),
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

    GEN_TR_a  = data_provider(TR, True, pip)
    GEN_TE_na = data_provider(TE, False, pip)
    GEN_TR_na = data_provider(TR, False, pip)

    # initialize model
    X, Y = next(GEN_TR_a)
    dataset_names = [d[0] for d in datasets_batch_padding]
    model = Resnet50DomainAdaptation(X, Y, dataset_names=dataset_names)
    model.summary()

    # define loss and optimizer
    n_domains = len(Y) - 1
    x_in = tf.placeholder(tf.float32, list(X.shape))
    y_predictions = [tf.placeholder(
        tf.float32, [args.batch, y.shape[1]]) for y in Y[:-1]]
    y_domain = tf.placeholder(tf.float32, [args.batch, Y[-1].shape[1]])
    y_predictions = model(x_in)
    prediction_losses = [mse(y_predictions[i], y_predictions[i])
                         for i in range(n_domains)]
    domain_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=y_predictions[-1], labels=y_domain)
    loss = tf.reduce_mean(tf.add_n(prediction_losses) + domain_loss)
    lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initialize variables and session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    K.set_session(sess)
    sess.run(init)

    # training 
    for e in range(args.epochs):
        for s in trange(args.steps_per_epoch, desc="Epoch {}".format(e)):
            X_batch, Y_batch = next(GEN_TR_na)            
            feed_dict = {
                x_in: X_batch,
                K.learning_phase(): 1,
                lr: 1e-3,
                y_domain: Y_batch[-1]}
            feed_dict.update(
                {y_predictions[i]: Y_batch[i] for i in range(n_domains)})
            K.set_value(model.get_layer('gradient_reversal_1').hp_lambda, 0.000001)
            _, l = sess.run([optimizer, loss], feed_dict)
    
    sys.exit()

    '''
    model.fit_generator(
        generator = GEN_TR_na, 
        steps_per_epoch = args.steps_per_epoch,
        epochs = args.epochs, 
        max_q_size = 100,
        validation_data = GEN_TE_na,
        validation_steps = 100,
        callbacks=[
            # save_predictions(GEN_TR_na, args.log_dir+'/TR_'),
            # save_predictions(GEN_TE_na, args.log_dir+'/TE_'),
            # save_best_model(GEN_TE_na,  args.log_dir),
            # keras.callbacks.ModelCheckpoint(
            #     args.log_dir+'/model.h5',save_weights_only=True),
            keras.callbacks.TensorBoard(log_dir=args.log_dir + '/Graph',
                                        write_graph=True)]
            )
    '''

    return model, pip
