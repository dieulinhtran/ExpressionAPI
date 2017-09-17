from tensorflow.contrib.keras import callbacks
import tensorflow.contrib.keras as K
from .evaluate import print_summary
import numpy as np
import os


class GradientReversalLambda(callbacks.Callback):
    def __init__(self, lambda_var, max_epoch, max_step):
        self.lambda_var = lambda_var
        self.max_epoch = max_epoch
        self.max_step = max_step
        self.epoch = 0
        self.step = 0


class save_predictions(callbacks.Callback):
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

class save_best_model(save_predictions):
    def __init__(self, *a, **kw):
        '''
        '''
        save_predictions.__init__(self, *a, **kw)
        self.best_score = -1e10


    def set_model(self, model):
        '''
        '''
        self.sess = K.backend.get_session()
        self.model  = model

    def on_epoch_end(self, epoch, logs={}):
        '''
        '''
        Y_hat = self.model.predict(self.X, batch_size=self.batch_size)
        if type(Y_hat) is not list: Y_hat=[Y_hat]
        corr = []
        for y0, y1 in zip(Y_hat, self.Y):

            y0 = y0[np.all(y1!=-1,1)]
            y1 = y1[np.all(y1!=-1,1)]
            corr_i = np.mean([np.corrcoef(i,j)[0,1] for i,j in zip(y0.T,y1.T)])
            corr.append(corr_i)
        current_score = np.mean(corr)

        if current_score >= self.best_score:
            self.best_score = current_score
            self.model.save_weights(self.log_dir+'/best_model.h5')
            with open(self.log_dir+'/best_model.txt', 'w') as f:
                print(epoch,self.best_score, file=f)
