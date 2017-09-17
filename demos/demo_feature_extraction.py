import ExpressionAPI
import glob
import h5py
from time import time
import numpy as np
from tqdm import tqdm

FE = ExpressionAPI.Feature_Extractor(verbose=0, batch_size=30)
out = './tmp/'


sequences = sorted(list(glob.glob('/homes/rw2614/data/databases/ck+/frames/*')))
# sequences = sorted(list(glob.glob('/homes/rw2614/data/raw/disfa/*')))

for i in tqdm(range(len(sequences))):
    seq = sequences[i]
    seq_frames = sorted(list(glob.glob(seq+'/*.png')))

    lab = []
    for frame in seq_frames:
        with open(frame[:-4]+'.csv','r') as f:
            content = f.readlines()
            dat = np.array([i[:-3].split(',')[1:] for i in content])
            lab.append( np.float32(dat) )
    lab = np.float32(lab)

    img, pts, pts_raw, cnn = FE.get_all_features_from_files(seq_frames)
    with h5py.File(out+seq.split('/')[-1]+'.h5') as tmp:
        tmp.create_dataset('pts', data=pts, dtype=np.float32)
        tmp.create_dataset('img', data=img, dtype=np.float32)
        tmp.create_dataset('lab', data=lab, dtype=np.float32)
        tmp.create_dataset('cnn', data=cnn, dtype=np.float32)
