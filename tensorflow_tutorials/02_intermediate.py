import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("TensorFlow Intermediate Tutorial")
print("===============================")

# 1. CNN for Image Classification
print("\n1. Convolutional Neural Network:")
# Create a simple CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("CNN Model Summary:")
cnn_model.summary()

# 2. RNN for Sequence Data
print("\n2. Recurrent Neural Network:")
# Create a simple RNN model
rnn_model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(None, 1)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

print("\nRNN Model Summary:")
rnn_model.summary()

# 3. Custom Layers
print("\n3. Custom Layer Implementation:")
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Test custom layer
custom_layer = CustomDense(3)
test_input = tf.ones((2, 4))
test_output = custom_layer(test_input)
print("\nCustom Layer Output Shape:", test_output.shape)

# 4. Custom Training Loop with Multiple Optimizers
print("\n4. Advanced Custom Training:")
# Create two small models
model_1 = models.Sequential([layers.Dense(2, input_shape=(2,))])
model_2 = models.Sequential([layers.Dense(2, input_shape=(2,))])

# Two optimizers
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer_2 = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def custom_train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:
        pred_1 = model_1(x)
        pred_2 = model_2(x)
        loss_1 = tf.reduce_mean(tf.square(pred_1 - y))
        loss_2 = tf.reduce_mean(tf.square(pred_2 - y))
    
    grads_1 = tape.gradient(loss_1, model_1.trainable_variables)
    grads_2 = tape.gradient(loss_2, model_2.trainable_variables)
    
    optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_variables))
    optimizer_2.apply_gradients(zip(grads_2, model_2.trainable_variables))
    
    return loss_1, loss_2

# 5. Custom Metrics
print("\n5. Custom Metrics Implementation:")
class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name='correct', initializer='zeros')
        self.total_predictions = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.equal(tf.cast(y_true, tf.int32), tf.cast(tf.argmax(y_pred, axis=1), tf.int32))
        values = tf.cast(values, tf.float32)
        if sample_weight is not None:
            values = tf.multiply(values, sample_weight)
        self.correct_predictions.assign_add(tf.reduce_sum(values))
        self.total_predictions.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.correct_predictions / self.total_predictions

# 6. Custom Callbacks
print("\n6. Custom Callback Implementation:")
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting epoch {epoch+1}")
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Finished batch {batch}, loss: {logs['loss']:.4f}")

# 7. Data Pipeline
print("\n7. Advanced Data Pipeline:")
# Create a complex data pipeline
def generate_data():
    for i in range(100):
        yield (np.random.randn(2), np.random.randint(0, 2))

dataset = tf.data.Dataset.from_generator(
    generate_data,
    output_signature=(
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

dataset = dataset.shuffle(buffer_size=1000)\
    .batch(32)\
    .prefetch(tf.data.AUTOTUNE)\
    .cache()

print("\nDataset structure:")
for x, y in dataset.take(1):
    print("Batch shape:", x.shape)

# 8. Model Subclassing
print("\n8. Model Subclassing Example:")
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

custom_model = CustomModel()
print("\nCustom Model Output Shape:", 
      custom_model(tf.random.normal((1, 20))).shape)

print("\nNote: These are intermediate-level concepts. Real applications would require proper data, hyperparameter tuning, and validation.")
