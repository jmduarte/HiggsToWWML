import keras
import numpy as np
import tables
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import math
import keras.backend as K
from models import dense

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def mean_absolute_relative_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

def mean_squared_relative_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.square((y_true - y_pred) / K.clip(K.square(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

def get_features_targets(file_name, features, targets):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    njets =  getattr(h5file.root,features[0]).shape[1]
    ntargets = len(targets)
    nfeatures = len(features)

    # allocate arrays
    feature_array = np.zeros((nevents,njets,nfeatures))
    target_array = np.zeros((nevents,njets,ntargets))

    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,:,i] = getattr(h5file.root,feat)[:]
    # load target arrays
    for (i, targ) in enumerate(targets):
        target_array[:,:,i] = getattr(h5file.root,targ)[:]

    feature_array = feature_array.reshape(nevents*njets,nfeatures)
    target_array = target_array.reshape(nevents*njets,ntargets)

    mask = ~np.all(feature_array == 0, axis=1)
    feature_array = feature_array[mask]
    target_array = target_array[mask]
    target_array[:,1] = np.clip(target_array[:,1],1e-6,None)
    
    h5file.close()
    return feature_array,target_array

def print_res(gen_mass, reco_mass, predict_mass, name='Mass_res.pdf'):
    #rel_err = (np.clip(predict_mass,1e-6,None) - np.clip(gen_mass,1e-6,None))/np.clip(gen_mass, 1e-6, None)
    plt.figure()          
    plt.hist(np.clip(gen_mass,1e-6,None), bins=np.linspace(0., 150., 50+1),alpha=0.5, label='generator mass')
    plt.hist(np.clip(reco_mass,1e-6,None), bins=np.linspace(0., 150., 50+1),alpha=0.5, label='soft drop mass')
    plt.hist(np.clip(predict_mass,1e-6,None), bins=np.linspace(0., 150., 50+1),alpha=0.5, label='predicted mass')
    plt.xlabel("Mass")
    plt.ylabel("Jets")
    plt.legend(loc='best')
    plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.tight_layout()
    plt.savefig(name)
    
def print_roc(y_true, y_predict, name = 'ROC.pdf'):
    # create ROC curve
    fpr, tpr, threshold = roc_curve(y_true, y_predict)
    acc = accuracy_score(y_true, y_predict>0.5)
    # plot ROC curve
    plt.figure()
    plt.plot(tpr, fpr, lw=2.5, label="AUC = {:.1f}%, Acc. = {:.1f}%".format(auc(fpr,tpr)*100.,acc*100.))
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.semilogy()
    plt.ylim(0.001,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.tight_layout()
    plt.savefig(name)
        
def main(args):
    file_path = 'data/raw/HWW.h5'
    features = ['CustomAK8Puppi_pt','CustomAK8Puppi_eta',
                'CustomAK8Puppi_phi','CustomAK8Puppi_msoftdrop']

    targets = ['isHiggs','LHEPart_mass']

    feature_array, target_array = get_features_targets(file_path, features, targets)
    nevents = feature_array.shape[0]
    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]

    keras_model = dense(nfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss=['binary_crossentropy', 'mean_squared_error'], 
                        loss_weights = [1., 0.0001])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    # fit keras model
    X = feature_array
    y = target_array

    fulllen = nevents
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    splits = [int(s) for s in splits]

    X_train = X[0:splits[0]]
    X_val = X[splits[1]:splits[2]]
    X_test = X[splits[0]:splits[1]]

    y_train = y[0:splits[0]]
    y_val = y[splits[1]:splits[2]]
    y_test = y[splits[0]:splits[1]]

    keras_model.fit(X_train, [y_train[:,:1], y_train[:,1:]], batch_size=1024, 
                    epochs=100, validation_data=(X_val, [y_val[:,:1], y_val[:,1:]]), shuffle=True,
                    callbacks = callbacks)

    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    predict_test = np.concatenate(predict_test,axis=1)

    print_res(y_test[:,1], X_test[:,-1], predict_test[:,1], name = 'Mass_res.pdf')
    print_roc(y_test[:,0], predict_test[:,0], name = 'ROC.pdf')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
