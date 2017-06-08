import glob
import os
import ExpressionAPI 

pwd = os.path.dirname(os.path.abspath(__file__))

img_files = sorted(list(glob.glob(pwd+'/tests/test_images/*.jpg')))


FE = ExpressionAPI.Feature_Extractor(output='flatten_1', verbose=1, batch_size=20)

FE.mod.summary()
# print the model details


img, pts, pts_raw, Z, AU_DISFA, AU_FERA = FE.get_all_features_from_files(img_files)
# img: input to the cnn
# pts: landmarks after transformation
# pts_raw: original landmarks
# Z: activations at flatten_1
# AU_DISFA: AU intensity levels for 12 AUs from the Disfa dataset
# AU_FERA : AU intensity levels for  5 AUs from the FERA  dataset


print('-'*80)
for f,au in zip(img_files, AU_DISFA):
    print(f.split('/')[-1])
    print('AU1:', au[0], 'AU4:', au[2], 'AU12:', au[6])
    print()
print('-'*80)

print('_tests_successful__')
