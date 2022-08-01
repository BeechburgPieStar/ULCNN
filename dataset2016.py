import pickle
import numpy as np

def load_data(filename = "RML2016.10a_dict.pkl",train_rate = 0.5):

    Xd = pickle.load(open(filename,'rb'),encoding='iso-8859-1')
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]

    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     #ndarray(1000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * train_rate)

    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))

    X_train = X[train_idx]
    X_test =  X[test_idx]

    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_train = np.argmax(Y_train, axis = 1)
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    Y_test = np.argmax(Y_test, axis = 1)


    return (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = load_data()
    np.save('train/x.npy',X_train)
    np.save('train/y.npy',Y_train)
    for snr in snrs:
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i=X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        np.save(f'test/x_snr={snr}.npy',test_X_i)
        np.save(f'test/y_snr={snr}.npy',test_Y_i)
