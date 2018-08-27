### Random Search
print(__doc__)

import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(units=neurons, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='TruncatedNormal', activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
    return(model)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=1)

# define the grid search parameters for batch size and epochs
batch_size = [10, 20, 40, 60]
epochs = [10, 25, 50, 100]
batch_epoch_grid = dict(batch_size=batch_size, epochs=epochs)

# define the grid search parameters for the optimizer
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
opt_grid = dict(optimizer=optimizer)

# define the grid search parameters for the initialization weights 
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
init_grid = dict(init_mode=init_mode)

# define the grid search parameters for learning rate and momentum
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learning_momentum_grid = dict(learn_rate=learn_rate, momentum=momentum)

# define the grid search parameters for neuron activation
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
activation_grid = dict(activation=activation)

# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
neuron_grid = dict(neurons=neurons)

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(model, param_distributions=init_grid,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, Y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)