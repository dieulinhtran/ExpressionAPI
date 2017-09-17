import os
import glob
import numpy as np

import tensorflow.contrib.keras as K
from skimage.io import imread

from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import layers
import FaceImageGenerator as FIG

_preprocess = FIG.image_pipeline.FACE_pipeline(
    histogram_normalization = False,
    output_size = [256, 256],
    face_size= 224,
    allignment_type = 'similarity'
    )

def get_input_features_from_numpy( img_batch):
    img, pts, pts_raw = _preprocess.batch_transform(img_batch, face_detect=True, preprocessing=False)
    img = np.float32(img)
    img_pp = img[:,16:-16,16:-16,:]
    img_pp -= np.apply_over_axes(np.mean, img_pp, [1,2])
    return img_pp, pts, pts_raw


def test_model(args):

    base_net = applications.resnet50.ResNet50( weights = 'imagenet' )

    inp_0 = base_net.input
    Z = base_net.get_layer('flatten_1').output

    out = []
    for n_outputs in [12,5]:
        net = layers.Dense( n_outputs )(Z)
        out.append( net )

    model = K.models.Model([inp_0], out)

    if args.weights=='default':
        weights = os.path.dirname(__file__)+'/models/ResNet50_aug_1.1/best_model.h5'
    else:
        weights = args.weights

    model.load_weights(weights)

    model.summary()
    files = sorted(list(glob.glob(args.input)))
    img_batch = np.stack([imread(i) for i in files])
    X = get_input_features_from_numpy( img_batch )
    pred = model.predict(X[0])
    if args.model=='disfa':
        title = ['file','AU1','AU2','AU4','AU5','AU6','AU9','AU12','AU15','AU17','AU20','AU25','AU26']
        AUs = pred[0]
    if args.model=='fera':
        title = ['file','AU6','AU10','AU12','AU14','AU17']
        AUs = pred[1]
    with open(args.output,'w') as f:
        for t in title:
            f.write(t)
            f.write(',')
        f.write('\n')
        for img_path, y in zip(files, AUs):
            items = [img_path]+list(y)
            for item in items:
                f.write(str(item))
                f.write(',')
            f.write('\n')
