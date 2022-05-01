import math
import tensorflow as tf
import keras
import numpy as np
import json

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
tf.keras.optimizers.Adam(learning_rate=40, beta_1=0.9, beta_2=0.999, epsilon=1e-09, amsgrad=False,name='Adam') 


model.compile(optimizer = 'adam', loss = "mean_squared_error")
xs = np.array(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), dtype = int)
ys = np.array([77, 79, 83, 86, 109, 111, 134, 159, 200, 241, 272, 283, 302, 355, 376], dtype = int)
model.fit(xs, ys, epochs = 160000)
for x in range(2023, 2031):
    z = model.predict([21.0])
    deaths = z*17500
    print(z)
    print(deaths)
    input()
