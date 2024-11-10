#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras import backend as K


# Check for GPU availability
def check_gpu():
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    return '/GPU:0'

# Load and preprocess image
def load_and_preprocess_image(image_path, img_size=400):
    image_pre = np.array(Image.open(image_path).resize((img_size, img_size)))
    image_pre = tf.constant(np.reshape(image_pre, ((1,) + image_pre.shape)))            # declare it as tensor
    print(image_pre.shape)
    plt.imshow(image_pre[0])
    plt.show()

    return image_pre 


# Define content and style loss functions
def compute_content_loss(content_output, generated_output):
    """ 
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer representing content of image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer representing content of image G

    Returns:
    J_content -- scalar 
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Unroll the 4D into a 3D matrix (as number of batch is always 1, will be like 2D ), [spatialsize, channels]
    a_C_unrolled = tf.reshape(a_C, shape=[1, n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[1, n_H*n_W, n_C])
    # Compute the total Cost (formula)
    J_content_cost = 1/(4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content_cost

def gram_matrix(A):
    return tf.matmul(A, A, transpose_b=True)

def compute_layer_style_cost(a_S, a_G):
    """ 
    Args:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations, represents style of image Style
    a_G -- tensor of dimension (1, n_H, n_W, n_c), hidden layer activations, represents style of image we want Generate
    
    Returns:
    J_style_cost -- tensor representing a scalar value of the hidden layer
    """

    # Retrieve shapes from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # we Unroll the hidden layers as in 'compute_content_cost' function
    a_S_unrolled = tf.reshape(a_S, shape=[1, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[1,n_H*n_W,n_C])
    # we compute the gram_matrices for bothh images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    # Finally we compute the style cost (formula)
    J_style_cost = (1/(4*n_C*n_C*(n_H*n_W)*(n_H*n_H))) * tf.reduce_sum(tf.square(tf.subtract(GS,GG)))

    return J_style_cost

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS):
    """ 
    Computes the overall style cost from the several chosen layers before

    Args:
    style_image_output -- our tensorflow model
    generated_image_output --

    Returns: 
    J_style -- tensor representing a scalar value.
    """
    J_style = 0
    # Set a_S as hidden layer activations from the layer selected
    a_S = style_image_output[:-1]
    # Set a_G as output of the choosen hidden  layer
    a_G = generated_image_output[:-1]
    # for every layer, compute the layer cost:
    for i, weigths in zip(range(len(a_S)), STYLE_LAYERS):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weigths[1] * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    return alpha * J_content + beta * J_style

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    return tf.keras.Model([vgg.input], outputs)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Training step function
@tf.function()
def train_step(generated_image, optimizer, vgg_model_output, a_C, a_S, STYLE_LAYERS, gpu_device):

    with tf.device(gpu_device):
        with tf.GradientTape() as tape:
            a_G = vgg_model_output(generated_image)
            J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS)
            J_content = compute_content_loss(a_C, a_G)
            J = total_cost(J_content, J_style, alpha=10, beta=40)
        grad = tape.gradient(J, [generated_image])
        optimizer.apply_gradients(zip(grad, [generated_image]))
        generated_image.assign(clip_0_1(generated_image))

    return J

def main(content_path, style_path, generated_path):
    # Define parameters and load images
    img_size = 400
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)
    ]
    gpu_device = check_gpu()

    content_image = Image.open(content_path)
    style_image = Image.open(style_path)
    

    content_image = load_and_preprocess_image(content_path, img_size)
    style_image = load_and_preprocess_image(style_path, img_size)

    vgg = tf.keras.applications.VGG19(include_top=False, 
                                      input_shape=(img_size, img_size, 3), 
                                      weights='pretrained_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False

    content_layer = [('block5_conv4'), 1]
    vgg_model_output = get_layer_outputs(vgg, STYLE_LAYERS + [content_layer])
    
    # Get content & style features
    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_output(preprocessed_content)

    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_output(preprocessed_style)  

    # Creating generated image
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(content_image), -0.25, 0.25)
    generated_image = tf.Variable(tf.add(generated_image, noise))
    generated_image = tf.Variable(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Run training
    epochs = 3000
    for i in range(epochs):
        J = train_step(generated_image, optimizer, vgg_model_output, a_C, a_S, STYLE_LAYERS, gpu_device)
        if i % 250 == 0:
            print(f"Epoch {i}, total cost: {J}")
        if i % 500 == 0:
            image = tensor_to_image(generated_image)
            plt.imshow(image)
            plt.show()

    output_path = f'generated_images/{generated_path}.jpg'
    image = tensor_to_image(generated_image)
    image.save(output_path)

    
    K.clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument('--content', type=str, required=True, help="Path to the content image")
    parser.add_argument('--style', type=str, required=True, help="Path to the style image")
    parser.add_argument('--output', type=str, required=True, help="Name of the output image (without extension)")

    args = parser.parse_args()
    main(args.content, args.style, args.output)
