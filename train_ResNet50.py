import ExpressionAPI
import argparse
import os
import pickle
from tensorflow.python.client import device_lib

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train ResNet50 for facial feature extraction')

    # input output
    parser.add_argument("-tr", "--trainingData", type=str, default='tr')
    parser.add_argument(
        "-l", "--log_dir", type=str,
        default='./ExpressionAPI/models/ResNet50_aug_1.1')

    # Augmentation
    parser.add_argument("-r", "--rotate", type=float, default=15)
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("-s", "--steps_per_epoch", type=int, default=2000)
    parser.add_argument("-g", "--gaussian_range", type=float, default=2)
    parser.add_argument("-n", "--normalization", type=int, default=0)
    parser.add_argument("-t", "--transform", type=float, default=0.05)
    parser.add_argument("-z", "--zoom", type=float, default=0.01)
    parser.add_argument("-b", "--batch", type=int, default=10) 

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    pickle.dump(args,open(args.log_dir+'/args.pkl','wb'))

    ExpressionAPI.train_model(args)
