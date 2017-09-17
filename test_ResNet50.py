import ExpressionAPI
import argparse
import os
import pickle

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Test ResNet50 for facial feature extraction')

    # input output
    parser.add_argument("-i","--input",   type=str, default='tests/test_images/S148_002/*.png')
    parser.add_argument("-o","--output",  type=str, default='tmp/test.csv')
    parser.add_argument("-m","--model",   type=str, default='disfa')
    parser.add_argument("-w","--weights", type=str, default='default')
    args = parser.parse_args()

    ExpressionAPI.test_model(args)
