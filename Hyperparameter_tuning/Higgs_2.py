from keras.layers import Dense, Activation, Dropout
from tensorflow.contrib.opt import PowerSignOptimizer
keras.optimizers.TFOptimizer(PowerSignOptimizer)
K.clear_session()
tb_call = keras.callbacks.TensorBoard(log_dir='C:\\Users\\tkalnik\\PycharmProjects\\RIIAA_Escuela18-master\\logs2', histogram_freq=0, write_graph=True, write_images=True)

model = Sequential()
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dense(units=20, input_dim=X_train.shape[-1], kernel_initializer='TruncatedNormal', activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='TruncatedNormal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=500, batch_size=32, callbacks=[tb_call])