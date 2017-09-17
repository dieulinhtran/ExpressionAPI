import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def _process(y_hat, y_lab, fun):
    '''
    - split y_true and y_pred in lists
    - removes frames where labels are unknown (-1)
    - returns list of predictions
    '''
    y1 = [x for x in y_hat.T]
    y2 = [x for x in y_lab.T]
    
    out = []
    for i, [_y1, _y2] in enumerate(zip(y1, y2)):
        idx = _y2!=-1
        _y1 = _y1[idx]
        _y2 = _y2[idx]
        if np.all(_y2==-1):
            out.append(np.nan)
        else:
            out.append(fun(_y1,_y2))
    return np.array(out)

def _acc(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.round(y_hat)
        y_lab = np.round(y_lab)
        return np.mean(y_hat==y_lab)
    return _process(y_hat, y_lab, fun)

def _mae(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.float32(y_hat)
        y_lab = np.float32(y_lab)
        return np.mean(np.abs(y_hat-y_lab))
    return _process(y_hat, y_lab, fun)

def _mse(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.float32(y_hat)
        y_lab = np.float32(y_lab)
        return np.mean((y_hat-y_lab)**2)
    return _process(y_hat, y_lab, fun)

def _rmse(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.float32(y_hat)
        y_lab = np.float32(y_lab)
        return (np.mean((y_hat-y_lab)**2))**0.5
    return _process(y_hat, y_lab, fun)

def _f1(y_hat, y_lab, threshold=0.5):
    def fun(y_hat,y_lab):
        y_hat = np.array(y_hat>=threshold)
        y_lab = np.array(y_lab>=threshold)
        tp = np.sum( (y_hat==1) * (y_lab==1) )
        fp = np.sum( (y_hat==1) * (y_lab==0) )
        fn = np.sum( (y_hat==0) * (y_lab==1) )
        if tp==0:
            return 0
        else:
            return (2*tp)/float(2*tp+fp+fn)
    return _process(y_hat, y_lab, fun)

def _icc(y_hat, y_lab, cas=3, typ=1):
    def fun(y_hat,y_lab):
        y_hat = y_hat[None,:]
        y_lab = y_lab[None,:]

        Y = np.array((y_lab, y_hat))
        # number of targets
        n = Y.shape[2]

        # mean per target
        mpt = np.mean(Y, 0)

        # print mpt.eval()
        mpr = np.mean(Y, 2)

        # print mpr.eval()
        tm = np.mean(mpt, 1)

        # within target sum sqrs
        WSS = np.sum((Y[0]-mpt)**2 + (Y[1]-mpt)**2, 1)

        # within mean sqrs
        WMS = WSS/n

        # between rater sum sqrs
        RSS = np.sum((mpr - tm)**2, 0) * n

        # between rater mean sqrs
        RMS = RSS

        # between target sum sqrs
        TM = np.tile(tm, (y_hat.shape[1], 1)).T
        BSS = np.sum((mpt - TM)**2, 1) * 2

        # between targets mean squares
        BMS = BSS / (n - 1)

        # residual sum of squares
        ESS = WSS - RSS

        # residual mean sqrs
        EMS = ESS / (n - 1)

        if cas == 1:
            if typ == 1:
                res = (BMS - WMS) / (BMS + WMS)
            if typ == 2:
                res = (BMS - WMS) / BMS
        if cas == 2:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
            if typ == 2:
                res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
        if cas == 3:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS)
            if typ == 2:
                res = (BMS - EMS) / BMS

        res = res[0]

        if np.isnan(res) or np.isinf(res):
            return 0
        else:
            return res
    return _process(y_hat, y_lab, fun)

def _pcc(y_hat, y_lab):
    def fun(y1, y2):
        res = np.corrcoef(y1, y2)[0, 1]
        if np.isnan(res) or np.isinf(res):
            return 0
        else:
            return res
    return _process(y_hat, y_lab, fun)

def print_summary(y_hat, y_lab, log_dir=None, verbose=0, mode='exp'):
    assert(y_hat.shape==y_lab.shape)

    # remove unlabeled frames
    idx = y_lab.reshape(y_lab.shape[0],-1).max(-1)>=0
    y_lab = y_lab[idx]
    y_hat = y_hat[idx]

    if y_hat.ndim==3:
        if mode=='exp':
            tmp = np.zeros(y_hat.shape[:2])
            for i in range(y_hat.shape[2]):
                tmp+=y_hat[:,:,i]*i
            y_hat = tmp

            tmp = np.zeros(y_lab.shape[:2])
            for i in range(y_lab.shape[2]):
                tmp+=y_lab[:,:,i]*i
            y_lab = tmp

        if mode=='max':
            y_hat = y_hat.argmax(2)
            y_lab = y_lab.argmax(2)

    data = []
    data.append(_icc(y_hat, y_lab))
    data.append(_pcc(y_hat, y_lab))
    data.append(_rmse(y_hat, y_lab))
    data.append(_mae(y_hat, y_lab))
    data.append(_acc(y_hat, y_lab))
    data.append(_f1(y_hat, y_lab))
    data = np.vstack(data)
    columns = [str(i) for i in np.arange(data.shape[1])]+['avr.']
    table = np.hstack((data,data.mean(1)[:,None]))
    index = ['ICC','PCC','RMSE','MAE','ACC','F1-b']
    t = pd.DataFrame(table, index=index, columns = columns)
    out = {
            'index':index,
            'columns':columns,
            'data':data,
            'table':t
            }
    if verbose:
        print(t)
        print()
        if log_dir:
            f = open(log_dir, 'w')
            print(t, file=f)
            f.close()
    return out

if __name__ == "__main__":
    import numpy as np
    np.random.seed(0)
    y_lab = np.random.randint(0,5,[100,4])
    y_hat = np.random.randint(0,5,[100,4])

    # out 0: same values
    y_lab[:,0] = y_hat[:,0]

    # out 1: every 3rd value is missing 
    y_lab[::3,1] = -1

    # out 2: all values is missing 
    y_lab[:,2] = -1

    # out 3: random values


    print('acc: ', _acc(y_hat,y_lab) )
    print('mae: ', _mae(y_hat,y_lab) )
    print('rmse:', _rmse(y_hat,y_lab))
    print('icc: ', _icc(y_hat,y_lab) )
    print('pcc: ', _pcc(y_hat,y_lab) )
    print('f1:  ', _f1(y_hat,y_lab)  )
    print_summary(y_hat, y_lab)
