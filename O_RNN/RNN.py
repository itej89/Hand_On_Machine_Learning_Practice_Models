import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import losses, metrics
from tensorflow.python.ops.gen_math_ops import mod
from tensorflow.python.ops.variables import model_variables

def generate_time_Series(batch_size, n_steps):
    import numpy as np
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    # print(offset1)
    # print(time)
    # print(len(time - offset1))
    series = 0.5 * np.sin((time - offset1) * (freq1 * 10  +10))
    series += 0.2 * np.sin((time - offset2) * (freq2 * 20  +20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


n_Steps = 50
series = generate_time_Series(10000, n_Steps+1)
X_train, y_train = series[:7000, :n_Steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_Steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_Steps], series[9000:, -1]


y_pred = X_valid[:, -1]
mse = np.mean(keras.losses.mean_squared_error(y_valid, y_pred))
print(f"Base line metric : {mse}")

#Sequntial model
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[50,1]),
#     keras.layers.Dense(1)
# ])

#RNN Model
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True ,input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer='adam', metrics=["MeanSquaredError"])

model.fit(X_train, y_train,
        batch_size=32,
        epochs=20,
        initial_epoch=0,
        validation_data=(X_valid, y_valid))

#forecasting next time step
y_pred = model.predict(X_test)
print(f"Next Step : {y_pred}")