import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import tensorflow_addons as tfa

print("TensorFlow Advanced Tutorial")
print("===========================")

# 1. Custom Training Loop with Multiple Models (GAN)
print("\n1. Generative Adversarial Network (GAN):")

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(128, input_shape=(100,)),
            layers.LeakyReLU(0.2),
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dense(784, activation='tanh')
        ])

    def call(self, inputs):
        return self.model(inputs)

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(256, input_shape=(784,)),
            layers.LeakyReLU(0.2),
            layers.Dense(128),
            layers.LeakyReLU(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)

# GAN Training
generator = Generator()
discriminator = Discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step_gan(images):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise)
        
        real_output = discriminator(images)
        fake_output = discriminator(generated_images)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                   cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# 2. Custom Layer with Complex Operations
print("\n2. Advanced Custom Layer:")
class ComplexDense(layers.Layer):
    def __init__(self, units, activation=None):
        super(ComplexDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.w_real = self.add_weight(
            name='w_real',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.w_imag = self.add_weight(
            name='w_imag',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        real_out = tf.matmul(inputs, self.w_real)
        imag_out = tf.matmul(inputs, self.w_imag)
        output = tf.complex(real_out, imag_out)
        if self.activation is not None:
            output = self.activation(output)
        return output + tf.cast(self.b, tf.complex64)

# 3. Custom Training Loop with Multiple Losses and Metrics
print("\n3. Advanced Training Loop:")
class MultiTaskModel(tf.keras.Model):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.shared = layers.Dense(64, activation='relu')
        self.task1 = layers.Dense(32, activation='relu')
        self.task2 = layers.Dense(16, activation='relu')
        self.output1 = layers.Dense(1)
        self.output2 = layers.Dense(1)
    
    def call(self, inputs):
        shared = self.shared(inputs)
        out1 = self.output1(self.task1(shared))
        out2 = self.output2(self.task2(shared))
        return out1, out2

@tf.function
def multi_task_train_step(model, x, y1, y2, optimizer):
    with tf.GradientTape() as tape:
        pred1, pred2 = model(x)
        loss1 = tf.reduce_mean(tf.square(y1 - pred1))
        loss2 = tf.reduce_mean(tf.square(y2 - pred2))
        total_loss = loss1 + loss2
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss1, loss2

# 4. Custom Training Loop with Gradient Accumulation
print("\n4. Gradient Accumulation:")
class GradientAccumulator:
    def __init__(self):
        self.gradients = []
        self.accumulation_steps = 0
    
    def accumulate(self, gradients):
        if not self.gradients:
            self.gradients = [tf.zeros_like(g) for g in gradients]
        
        for i, g in enumerate(gradients):
            self.gradients[i] += g
        self.accumulation_steps += 1
    
    def apply_and_reset(self, optimizer, variables):
        for g, v in zip(self.gradients, variables):
            g = g / tf.cast(self.accumulation_steps, g.dtype)
        optimizer.apply_gradients(zip(self.gradients, variables))
        self.gradients = []
        self.accumulation_steps = 0

# 5. Custom Regularizers and Constraints
print("\n5. Advanced Regularization:")
class ComplexRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2
    
    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(tf.complex(
                tf.real(x), tf.imag(x))))
        if self.l2:
            regularization += self.l2 * tf.reduce_sum(tf.square(tf.abs(
                tf.complex(tf.real(x), tf.imag(x)))))
        return regularization

# 6. Custom Training Loop with Learning Rate Scheduling
print("\n6. Advanced Learning Rate Scheduling:")
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps):
        super(CosineAnnealingSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * step / self.decay_steps))
        return self.initial_learning_rate * cosine_decay

# 7. Custom Metrics with Multiple States
print("\n7. Advanced Metrics:")
class MultiStateMetric(tf.keras.metrics.Metric):
    def __init__(self, name='multi_state_metric', **kwargs):
        super(MultiStateMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight('tp', initializer='zeros')
        self.false_positives = self.add_weight('fp', initializer='zeros')
        self.true_negatives = self.add_weight('tn', initializer='zeros')
        self.false_negatives = self.add_weight('fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        
        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), self.dtype)))
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), self.dtype)))
        self.true_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), 
                                               tf.logical_not(y_pred)), self.dtype)))
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), self.dtype)))
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return 2 * ((precision * recall) / (precision + recall))

print("\nNote: These are advanced concepts that require good understanding of TensorFlow internals and machine learning principles.")
