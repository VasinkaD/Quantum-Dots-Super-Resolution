import numpy as np
import tensorflow as tf

##########################################
##########################################

# Loss and Metric functions

def Gauss_function(x, sigma):
    Gauss_amplitude = np.exp(-(x)**2/(2*sigma**2))       #Amplitude is the Gaussian profile
    Gauss_intensity = Gauss_amplitude**2                 #Intesity is also Gaussian but of different width
    return Gauss_intensity

def Gauss_kernel(c):
    radius = int(np.ceil(3*c))
    k = int(2*radius+1)
    
    x = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx = np.sqrt(x**2 + np.transpose(x)**2)                                         #Field for kernel
    
    unnormed_psf_matrix = Gauss_function(xx, c)                                     #Gaussian kernel; size based on 3sigma rule of 99.7%
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Gaussian kernel
    
    return normed_psf_matrix

def Custom_metric_func(y_true, y_pred):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Gauss_kernel(2)[:,:,None,None]
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Valid padding, not to lose information on the borders
    y_true_conv = tf.nn.conv2d(y_true, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    absolute_difference = tf.math.abs(y_true_conv - y_pred_conv)
    return tf.reduce_sum(absolute_difference, axis=(-1,-2,-3))

def Custom_loss_func(y_true, y_pred):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Gauss_kernel(2)[:,:,None,None]
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Valid padding, not to lose information on the borders
    y_true_conv = tf.nn.conv2d(y_true, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    squared_difference = tf.square(y_true_conv - y_pred_conv)
    loss = tf.reduce_sum(squared_difference, axis=(-1,-2,-3))
    
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=(-1,-2,-3))
    
    return loss + entropy * 5e-5

##########################################

# Model functions

def Load_CNN_model(path):
    CNN_model = tf.keras.models.load_model(path, custom_objects={
        "Custom_mse_conv_func_strong_regularized": Custom_loss_func, 
        "Custom_mae_conv_func": Custom_metric_func})
    return CNN_model

def Normalize_CNN_input(data):
    if len(data.shape) == 2:
        data = data[None,:]
    
    data_subbed = data - data.min(axis=(-1,-2))[:,None,None]
    data_normed = data_subbed / data_subbed.sum(axis=(-1,-2))[:,None,None]
    data_in = data_normed[:,:,:,None]
    
    return data_in

def CNN_predict(model, data):
    data_in = Normalize_CNN_input(data)
    predicted = model.predict(data_in).squeeze()
    return predicted













