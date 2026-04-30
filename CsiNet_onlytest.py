"""
CsiNet Inference Only for CSI Compression and Reconstruction
This script loads a pre-trained CsiNet autoencoder model (architecture + weights)
to perform only inference on test CSI data for indoor/outdoor wireless environments.
It evaluates model performance with NMSE (Normalized Mean Square Error) and correlation coefficient,
measures per-sample inference time, and visualizes the original vs. reconstructed CSI amplitude.
No model training is performed in this script.
"""
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model, model_from_json
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
tf.compat.v1.reset_default_graph()
# ────────────────────────  Environment & CSI Configuration  ───────────────────────── #
envir = 'indoor' #'indoor' or 'outdoor' -> Select the wireless propagation environment
# ────────────────────────  CSI Image Parameters  ────────────────────────────── #
img_height = 32        # CSI matrix height (spatial dimension)
img_width = 32         # CSI matrix width (frequency dimension)
img_channels = 2       # Real and imaginary parts of CSI (2 separate channels)
img_total = img_height*img_width*img_channels  # Total flattened CSI feature dimensions
# ────────────────────────  Network Hyperparameters  ─────────────────────────── #
residual_num = 2       # Number of residual blocks in the decoder (match pre-trained model)
encoded_dim = 512      # Compression dimension (match pre-trained model): 1/4→512,1/16→128,1/32→64,1/64→32
# Generate model file name (consistent with training script naming convention)
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)
# ────────────────────────  Load Pre-trained CsiNet Model  ────────────────────── #
outfile_json = "saved_model/model_%s.json" % file
with open(outfile_json, 'r') as json_file:
    loaded_model_json = json_file.read()

autoencoder = model_from_json(loaded_model_json)

outfile_weights = "saved_model/model_%s.weights.h5" % file
autoencoder.load_weights(outfile_weights)
# ────────────────────────  Test Data Loading and Preprocessing  ─────────────── #
# Load MATLAB-formatted test CSI data for the selected environment
if envir == 'indoor':
    mat = sio.loadmat('data/dataset_4.mat') 
    x_test = mat['H_test']
elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT'] # Outdoor test CSI data array

# Convert data to float32 (matching training data type for inference)
x_test = x_test.astype('float32')
# Reshape to channels_first format: [batch_size, channels, height, width] (match model input)
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
# ────────────────────────  CSI Reconstruction Inference  ────────────────────── #
# Measure total inference time for the entire test set
tStart = time.time()
x_hat = autoencoder.predict(x_test)  # Reconstruct CSI from test input (forward pass only)
tEnd = time.time()
# Calculate and print average inference time per test sample
print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))
# ────────────────────────  Performance Evaluation (NMSE ONLY)  ──────────────────── #
# Convert reconstructed/raw CSI from [0,1] to complex domain (-0.5~0.5 for real/imag)
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)  # Raw complex CSI

x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)  # Reconstructed complex CSI

# Compute NMSE (in dB) for CSI reconstruction
power = np.sum(abs(x_test_C)**2, axis=1)    # Power of original complex CSI
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)  # MSE between original and reconstructed complex CSI

# Print key performance metrics
print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))  # NMSE in logarithmic dB scale
# ────────────────────────  CSI Reconstruction Visualization  ────────────────── #
import matplotlib.pyplot as plt
'''Plot absolute amplitude of original and reconstructed complex CSI (first 10 test samples)'''
n = 10  # Number of samples to visualize
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original CSI absolute amplitude
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)  # Invert for better visualization
    plt.gray()  # Grayscale colormap
    ax.get_xaxis().set_visible(False)  # Hide x-axis
    ax.get_yaxis().set_visible(False)  # Hide y-axis
    ax.invert_yaxis()  # Invert y-axis for consistent spatial alignment

    # Display reconstructed CSI absolute amplitude
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)  # Invert for better visualization
    plt.gray()  # Grayscale colormap
    ax.get_xaxis().set_visible(False)  # Hide x-axis
    ax.get_yaxis().set_visible(False)  # Hide y-axis
    ax.invert_yaxis()  # Invert y-axis for consistent spatial alignment
plt.show()  # Show the visualization plot