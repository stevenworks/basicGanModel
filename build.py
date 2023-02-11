# Tensorflow and Keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Global variable initialization
BATCH_SIZE = 100
EPOCHS = 10_000
Z_DIM = 100

# Visualization noise
z_vis = tf.random.normal([10, Z_DIM])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_iter = iter(tf.data.Dataset.from_tensor_slices(x_train).shuffle(4 * BATCH_SIZE).batch(BATCH_SIZE).repeat())

# Plot dataset
for id in range(25):
  # Define subplot with axis off
  plt.subplot(5, 5, id + 1)
  plt.axis("off")
  # Plot raw pixel data
  plt.imshow(x_train[id], cmap="gray_r")
plt.show()

# Function for building discriminator model
def build_discriminator(input_shape=(28, 28, 1), layers=3, verbose=True):
  """
  Utility function to build a CNN discriminator
  Parameters:
    input_shape:
      purpose: shape of input image
      type: tuple
      default: (28, 28, 1) -> MNIST 
    layers:
      purpose: number of dense layers
      type: int
      default: 3
    verbose:
      purpose: print model summary
      type: bool
      default: True
   Return:
    discriminator:
      type: tf.keras.Model
  """
  
  # Sequential model
  discriminator = tf.keras.models.Sequential()
  
  # Convolutional layers
  discriminator.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", input_shape = input_shape))
  discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  discriminator.add(tf.keras.layers.Dropout(0.1))
  discriminator.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
  discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  discriminator.add(tf.keras.layers.Dropout(0.2))
  
  # Base
  discriminator.add(tf.keras.layers.Flatten())
  for id in range(layers):
    discriminator.add(tf.keras.layers.Dense(512, activation="tanh"))
    discriminator.add(tf.keras.layers.Dropout(0.2))
  discriminator.add(tf.keras.layers.Dense(1))
  
  # Model summary
  if verbose:
    discriminator.summary()
    
  return discriminator
  
# Function for building generator model
def build_generator(z_dim=100, output_shape=(28, 28), verbose=True):
  """
  Utility function to build a MLP generator
  Parameters:
    z_dim:
      purpose: shape of input noise vector
      type: int
      default: 100
    output_shape:
      purpose: shape of output image
      type: tuple
      default: (28, 28) -> MNIST
    verbose:
      purpose: print model summary
      type: bool
      default: True
   Return:
    generator:
      type: tf.keras.Model
  """
  
  # Sequential model
  generator = tf.keras.models.Sequential()
  
  generator.add(tf.keras.layers.Input(shape=(z_dim,)))
  generator.add(tf.keras.layers.Dense(256, input_dim=z_dim))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(512))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(1024))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(1024))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(2048))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(4096))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(8192))
  generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))
  generator.add(tf.keras.layers.Dense(np.prod(output_shape), activation="tanh"))
  generator.add(tf.keras.layers.Reshape(output_shape))
  
  # Model summary 
  if verbose:
    generator.summary()
    
  return generator
  

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)

# Loss functions
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(discriminator, fake_images):
  return cross_entropy_loss(tf.ones_like(discriminator(fake_images)), discriminator(fake_images))
def discriminator_loss(discriminator, real_images, fake_images):
  return cross_entropy_loss(tf.ones_like(discriminator(real_images)), discriminator(real_images)) + cross_entropy_loss(tf.zeros_like(discriminator(fake_images)), discriminator(fake_images))
  
# Training loop
for epoch in range(EPOCHS):
  z_batch = tf.random.normal([BATCH_SIZE, Z_DIM]) # Produce input noise
  real_images = next(x_iter) # Fetch real images
  
  # Find gradients
  with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
    fake_images = generator(z_batch)
    generator_curr_loss = generator_loss(discriminator, fake_images)
    discriminator_curr_loss = discriminator_loss(discriminator, real_images, fake_images)
  generator_gradient = generator_tape.gradient(generator_curr_loss, generator.trainable_variables)
  discriminator_gradient = discriminator_tape.gradient(discriminator_curr_loss, discriminator.trainable_variables)
  
  # Apply gradients
  generator_opt.apply_gradients(zip(generator_gradient, generator.trainable_variables))
  discriminator_opt.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
  
  if epoch % 100 == 0: # Each hundred epoch, log error and plot visualization
    print(f"At epoch {epoch+1:,}, generator loss: {generator_curr_loss:.6f}, discriminator loss: {discriminator_curr_loss:.6f}")
    for id in range(10):
      plt.subplot(2, 5, id+1)
      plt.imshow(generator(z_vis)[id,:,:]*255.0)
      plt.axis("off")
    plt.show()
      
          
  





