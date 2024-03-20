from Concat import Concat
from ConvolutionalLayer import ConvolutionalLayer
from MaxPoolLayer import MaxPoolLayer
from UpconvolutionLayer import UpconvolutionalLayer
from UpsamplingLayer import UpsamplingLayer
from SquaredError import SquaredError
from halfLayers import halfLayers
import matplotlib.pyplot as plt
import numpy as np
import os

# UNET LAYERS ARCHITECTURE
Down_conv_1 = ConvolutionalLayer(kwidth=2, kcount=2)
Max_pool_1 = MaxPoolLayer(2, 2)
Down_conv_2 = ConvolutionalLayer(kwidth=2, kcount=2)
Max_pool_2 = MaxPoolLayer(2, 2)
Down_conv_3 = ConvolutionalLayer(kwidth=2, kcount=2)
Max_pool_3 = MaxPoolLayer(2, 2)
Down_conv_4 = ConvolutionalLayer(kwidth=2, kcount=2)
Up_sample_1 = UpsamplingLayer(2)
Half_1 = halfLayers()
Concat_1 = Concat(Down_conv_3)
Half_2 = halfLayers()
Up_conv_1 = UpconvolutionalLayer(3, 1)
Up_sample_2 = UpsamplingLayer(2)
Half_3 = halfLayers()
Concat_2 = Concat(Down_conv_2)
Half_4 = halfLayers()
Up_conv_2 = UpconvolutionalLayer(3, 1)
Up_sample_3 = UpsamplingLayer(2)
Half_5 = halfLayers()
Concat_3 = Concat(Down_conv_1)
Up_conv_3 = UpconvolutionalLayer(3, 1)
Half_6 = halfLayers()
Half_7 = halfLayers()
Up_conv_4 = UpconvolutionalLayer(3, 1)
Squared_Error = SquaredError()

def forward_propagation_unet(input_image):
    forward_down_conv_1 = Down_conv_1.forward(input_image)
    forward_max_pool_1 = Max_pool_1.forward(forward_down_conv_1)
    forward_down_conv_2 = Down_conv_2.forward(forward_max_pool_1)
    forward_max_pool_2 = Max_pool_2.forward(forward_down_conv_2)
    forward_down_conv_3 = Down_conv_3.forward(forward_max_pool_2)
    forward_max_pool_3 = Max_pool_3.forward(forward_down_conv_3)
    forward_down_conv_4 = Down_conv_4.forward(forward_max_pool_3)
    forward_up_sample_1 = Up_sample_1.forward(forward_down_conv_4)
    forward_half_layer_1 = Half_1.forward(forward_up_sample_1)
    forward_concat_1 = Concat_1.forward(forward_half_layer_1)
    forward_half_layer_2 = Half_2.forward(forward_concat_1)
    forward_up_conv_1 = Up_conv_1.forward(forward_half_layer_2)
    forward_up_sample_2 = Up_sample_2.forward(forward_up_conv_1)
    forward_half_layer_3 = Half_3.forward(forward_up_sample_2)
    forward_concat_2 = Concat_2.forward(forward_half_layer_3)
    forward_half_layer_4 = Half_4.forward(forward_concat_2)
    forward_up_conv_2 = Up_conv_2.forward(forward_half_layer_4)
    forward_up_sample_3 = Up_sample_3.forward(forward_up_conv_2)
    forward_half_layer_5 = Half_5.forward(forward_up_sample_3)
    forward_concat_3 = Concat_3.forward(forward_half_layer_5)
    forward_half_layer_6 = Half_6.forward(forward_concat_3)
    forward_up_conv_3 = Up_conv_3.forward(forward_half_layer_6)
    forward_half_layer_7 = Half_7.forward(forward_up_conv_3)
    forward_up_conv_4 = Up_conv_4.forward(forward_half_layer_7)
    return forward_up_conv_4

def backward_propagation_unet(Y, Yhat):
    gradient = Squared_Error.gradient(Y, Yhat)
    plt.imshow(gradient.reshape(64,64), cmap='viridis')
    plt.show()
    print(Squared_Error.eval(Y, Yhat))
    gradient = gradient.reshape((1, 64, 64))
    backward_up_conv_4 = Up_conv_4.backward(gradient)
    Up_conv_4.updateWeights(gradient, 0.01)
    backward_half_layer_7 = Half_7.backward(backward_up_conv_4)
    backward_up_conv_3 = Up_conv_3.backward(backward_half_layer_7)
    Up_conv_3.updateWeights(backward_half_layer_7, 0.01)
    backward_half_layer_6 = Half_6.backward(backward_up_conv_3)
    backward_concat_3 = Concat_3.backward(backward_half_layer_6)
    backward_half_layer_5 = Half_5.backward(backward_concat_3)
    backward_up_sample_3 = Up_sample_3.backward(backward_half_layer_5)
    backward_up_conv_2 = Up_conv_2.backward(backward_up_sample_3)
    Up_conv_2.updateWeights(backward_up_sample_3, 0.01)
    backward_half_layer_4 = Half_4.backward(backward_up_conv_2)
    backward_concat_2 = Concat_2.backward(backward_half_layer_4)
    backward_half_layer_3 = Half_3.backward(backward_concat_2)
    backward_up_sample_2 = Up_sample_2.backward(backward_half_layer_3)
    backward_up_conv_1 = Up_conv_1.backward(backward_up_sample_2)
    Up_conv_1.updateWeights(backward_up_sample_2, 0.01)
    backward_half_layer_2 = Half_2.backward(backward_up_conv_1)
    backward_concat_1 = Concat_1.backward(backward_half_layer_2)
    backward_half_layer_1 = Half_1.backward(backward_concat_1)
    backward_up_sample_1 = Up_sample_1.backward(backward_half_layer_1)
    backward_down_conv_4 = Down_conv_4.backward(backward_up_sample_1)
    Down_conv_4.updateWeights(backward_up_sample_1, 0.01)
    backward_max_pool_3 = Max_pool_3.backward(backward_down_conv_4)
    backward_down_conv_3 = Down_conv_3.backward(backward_max_pool_3)
    Down_conv_3.updateWeights(backward_max_pool_3, 0.01)
    backward_max_pool_2 = Max_pool_2.backward(backward_down_conv_3)
    backward_down_conv_2 = Down_conv_2.backward(backward_max_pool_2)
    Down_conv_2.updateWeights(backward_max_pool_2, 0.01)
    backward_max_pool_1 = Max_pool_1.backward(backward_down_conv_2)
    backward_down_conv_1 = Down_conv_1.backward(backward_max_pool_1)
    Down_conv_1.updateWeights(backward_max_pool_1, 0.01)
    return backward_down_conv_1

if __name__ == "__main__":
    # Getting data
    unlensed = []
    # Directory containing the .npy files
    directory = r"lensingData/Data/train/Lensed"

    # Get a list of .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')][:20]
    # Sort the files to maintain order
    npy_files.sort()
    # Iterate through the list and plot each file
    for idx, npy_file in enumerate(npy_files):
        # Load the .npy file
        data = np.load(os.path.join(directory, npy_file))
        unlensed.append(data)

    unlensed = np.array(unlensed)
    lensed = []
    # Directory containing the .npy files
    directory = r"lensingData/Data/train/Unlensed"

    # Get a list of .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')][:20]

    # Sort the files to maintain order
    npy_files.sort()

    # Iterate through the list and plot each file
    for idx, npy_file in enumerate(npy_files):
        # Load the .npy file
        data = np.load(os.path.join(directory, npy_file))
        lensed.append(data)

    lensed = np.array(lensed)


    first_image = np.array([unlensed[0]])
    forward_down_conv_1 = Down_conv_1.forward(first_image)
    first_image.shape, forward_down_conv_1.shape
    plt.imshow(first_image[0], cmap='viridis')
    plt.show()
    for i in range(20):
        Yhat = forward_propagation_unet(first_image)
        flatten_Yhat = Yhat.flatten()
        first_image = backward_propagation_unet(np.array(lensed[0]).flatten(), flatten_Yhat)

    plt.imshow(first_image[0], cmap='viridis')
    plt.show()
