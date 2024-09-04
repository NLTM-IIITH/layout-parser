from scipy.stats import gaussian_kde
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# kernel_bandwidth = 
def kde_estimate(data):
    # data = np.array(data)
    
    # if data.ndim == 1:
    #     d = 1
    # else:
    #     d = data.shape[1]
        
    n = len(data)
    d = np.array(data).ndim
    
    # kernel_bandwidth = n**(-1./(d+4))
    kernel_bandwidth = 0.1
    print('Kernel Bandwidth:', kernel_bandwidth)
    kde = gaussian_kde(data, bw_method=kernel_bandwidth)
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde.evaluate(x)
    print('KDE VALUES')
    print(type(kde_values),kde_values.size,sorted(kde_values)[0])
    peak_index = np.argmax(kde_values)
    peak_value = x[peak_index]

    return math.ceil(peak_value)

    # return x,kde_values, math.ceil(peak_value)
    # return kde

def kde_estimate_d(data,direction):
    data = np.array(data)
    
    if data.ndim == 1:
        d = 1
    else:
        d = data.shape[1]
        
    n = len(data)
    d = np.array(data).ndim
    if direction == 'horizontal':
        kernel_bandwidth = n**(-1./(d+4))
    # kernel_bandwidth = n**(-1./(d+4))
    elif direction == 'vertical':
        kernel_bandwidth = 0.1
    print('Kernel Bandwidth:', kernel_bandwidth)
    kde = gaussian_kde(data, bw_method=kernel_bandwidth)
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde.evaluate(x)
    print('KDE VALUES')
    print(type(kde_values),kde_values.size,sorted(kde_values)[0])
    peak_index = np.argmax(kde_values)
    peak_value = x[peak_index]

    return math.ceil(peak_value)


# def kde_para(data):
#     n = len(data)
#     d = np.array(data).ndim
#     kernel_bandwidth = n**(-1./(d+4))
#     # kernel_bandwidth = 0.1
#     kde = gaussian_kde(data, bw_method=kernel_bandwidth)
#     x = np.linspace(min(data), max(data), 1000)
#     kde_values = kde.evaluate(x)
#     # peak_index = np.argmax(kde_values)
#     # peak_value = x[peak_index]
#     peak_index = sorted(list(kde_values)).index(sorted(list(kde_values))[-2])
#     peak_value = x[peak_index]
#     return math.ceil(peak_value)


import numpy as np
from scipy.stats import gaussian_kde
import math
from scipy.signal import find_peaks

def kde_para(data):
    n = len(data)
    d = np.array(data).ndim
    kernel_bandwidth = n**(-1./(d+4))
    kde = gaussian_kde(data, bw_method=kernel_bandwidth)
    
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde.evaluate(x)
    
    peaks, _ = find_peaks(kde_values)
    
    if len(peaks) < 2:
        # If there is only one peak, return that peak value
        peak_value = x[peaks[0]]
    else:
        # Sort the peaks by their kde_values in descending order and get the second peak
        sorted_peaks = peaks[np.argsort(kde_values[peaks])][::-1]
        peak_value = x[sorted_peaks[1]]
    
    return math.ceil(peak_value)

# def plot_kde(data, direction, image_file_name):
#         n = len(data)
#         d = np.array(data).ndim
#         kernel_bandwidth = n**(-1./(d+4))
#         # kernel_bandwidth = 0.1
#         kde = gaussian_kde(data, bw_method=kernel_bandwidth)
#         x = np.linspace(min(data), max(data), 1000)
#         kde_values = kde.evaluate(x)
#         # peak_index = np.argmax(kde_values)
#         # peak_value = x[peak_index]
#         peak_index = sorted(list(kde_values)).index(sorted(list(kde_values))[-2])
#         peak_value = x[peak_index]
#         plt.plot(x, kde_values)
#         plt.title(f'{direction} neighbors')
#         plt.xlabel('Distance')
#         plt.ylabel('Density')
#         plt.axvline(x=peak_value, color='r', linestyle='--')
#         os.makedirs('./kde_plots', exist_ok=True)
#         plt.savefig(f'./kde_plots/{image_file_name}_{direction}_neighbors.png')
#         plt.close()


def plot_kde(data, direction, image_file_name):
    n = len(data)
    d = np.array(data).ndim
    kernel_bandwidth = n**(-1./(d+4))
    kde = gaussian_kde(data, bw_method=kernel_bandwidth)
    
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde.evaluate(x)
    
    peaks, _ = find_peaks(kde_values)
    
    if len(peaks) < 2:
        # If there is only one peak, return that peak value
        peak_value = x[peaks[0]]
    else:
        # Sort the peaks by their kde_values in descending order and get the second peak
        sorted_peaks = peaks[np.argsort(kde_values[peaks])][::-1]
        peak_value = x[sorted_peaks[1]]
    plt.plot(x, kde_values)
    plt.title(f'{direction} neighbors')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.axvline(x=peak_value, color='r', linestyle='--')
    os.makedirs('./kde_plots', exist_ok=True)
    plt.savefig(f'./kde_plots/{image_file_name}_{direction}_neighbors.png')
    plt.close()