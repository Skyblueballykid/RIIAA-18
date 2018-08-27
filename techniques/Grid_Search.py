### Grid Search

import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(units=100, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
    model.add(Dense(units=100, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
    model.add(Dense(units=100, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
    model.add(Dense(units=100, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
    model.add(Dense(units=100, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
    model.add(Dense(units=100, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
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
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 25, 50, 100, 125, 150]
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

# Interchange the param-grid option for the appropriate parameters to search
grid = GridSearchCV(estimator=model, param_grid=batch_epoch_grid, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))