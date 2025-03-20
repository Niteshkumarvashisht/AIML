import tensorflow as tf

print("cuDNN Intermediate Tutorial")
print("===========================")

# 1. Using cuDNN for RNNs
print("\n1. RNN with cuDNN:")
model = tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(64, return_sequences=True, input_shape=(10, 16)),
    tf.keras.layers.CuDNNLSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# 2. Using cuDNN for GRUs
print("\n2. GRU with cuDNN:")
model_gru = tf.keras.Sequential([
    tf.keras.layers.CuDNNGRU(64, return_sequences=True, input_shape=(10, 16)),
    tf.keras.layers.CuDNNGRU(32),
    tf.keras.layers.Dense(1)
])
model_gru.compile(optimizer='adam', loss='mse')
model_gru.summary()
