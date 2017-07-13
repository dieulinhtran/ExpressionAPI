from tensorflow.contrib.keras import callbacks
import tensorflow.contrib.keras as K
from .evaluate import print_summary
import numpy as np
import os

class save_predictions(object):
    def __init__(self, data_generator, log_dir, nb_batches=500, batch_size=10):
        '''
        '''
        self.batch_size = batch_size
        self.log_dir = log_dir

        # generate data from N batchs
        X, Y = [], []
        for i in range(nb_batches):
            x, y = next(data_generator)
            X.append(x)
            Y.append(y)

        # transpose and concatonate datasets
        Y = list(map(list, zip(*Y)))
        Y = [np.vstack(i) for i in Y]

        X = list(map(list, zip(*X)))
        X = [np.vstack(i) for i in X]

        self.X = X
        self.Y = Y

    def set_model(self, model):
        '''
        '''
        self.sess = K.backend.get_session()
        self.model  = model

    def on_epoch_end(self, epoch, logs={}):
        '''
        '''
        Y_hat = self.model.predict(self.X, batch_size=self.batch_size)

        model_loss = self.model.evaluate(self.X, self.Y, verbose=0)
        # model_loss is an scalar if there is only a single output 
        # however, it is a list for models with multi outputs where the first element is the average loss!
        # this applies also to the predictions: Y_hat
        try:
            # get losses from each output
            losses = model_loss[1:]
        except IndexError:
            # get loss from single output
            losses = [model_loss]
            Y_hat = [Y_hat]

        for i, [loss, y0, y1] in enumerate(zip(losses, Y_hat, self.Y)):
            path = self.log_dir+str(i).zfill(2)
            summary = print_summary(y0, y1)
            with open(path+'_summary.txt','a') as f:
                print('\n', file=f)
                print('epoch: '+str(epoch).zfill(5)+'\n', file=f)
                print(summary['table'], file=f)


            np.save(self.log_dir+str(i).zfill(2), {'y_hat':y0, 'y_lab':y1})

            index  = ['loss'] + list(summary['table'].index)
            values = np.hstack([[loss],summary['table'].values[:,-1]])

            if os.path.isfile(path+'.csv') == False:
                with open(path+'.csv', 'a') as f:
                    f.write(','.join(index)+'\n')

            with open(path+'.csv', 'a') as f:
                f.write(','.join([str(v) for v in values])+'\n')
