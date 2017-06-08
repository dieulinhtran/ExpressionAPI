import pickle
import glob
import os
from ExpressionAPI.visualize import make_hub 
from skimage.io import imread, imsave

pwd = os.path.dirname(os.path.abspath(__file__))

img_files = sorted(list(glob.glob(pwd+'/tests/test_images/*.jpg')))
img, pts, pts_raw, Z, AU_DISFA, AU_FERA = pickle.load(open('./test_out.pkz','rb'))
img_raw = [imread(i) for i in img_files]

f = 0
for f in range(len(img_files)):
    img_out = make_hub(img_raw[f], pts_raw[f], AU_DISFA[f], img[f])
    imsave('out.png', img_out)
