import matplotlib
matplotlib.use('TkAgg')
import datetime as DT
import os
from matplotlib import pyplot as plt
import geoprocess
import yellowfinLib
import numpy as np
from scipy import signal
import cv2
# from yellowfinLib import find_bottom_percentile

########################################################################################
def two_param_bivariate_filter_1d(signal, spatial_sigma, intensity_sigma):
    """
    Apply a 1D bivariate (bilateral-like) filter to a 1D signal.

    Parameters:
        signal (np.ndarray): Input 1D signal.
        spatial_sigma (float): Spatial standard deviation.
        intensity_sigma (float): Intensity standard deviation.

    Returns:
        np.ndarray: Filtered 1D signal.
    """
    length = len(signal)
    filtered_signal = np.zeros_like(signal)

    for i in range(length):
        weights = np.zeros(length)

        for j in range(length):
            # Spatial weight
            spatial_weight = np.exp(-((i - j) ** 2) / (2 * spatial_sigma ** 2))

            # Intensity weight
            intensity_weight = np.exp(-((signal[i] - signal[j]) ** 2) / (2 * intensity_sigma ** 2))

            # Combined weight
            weights[j] = spatial_weight * intensity_weight

        # Normalize weights
        weights /= weights.sum()

        # Apply weights to filter
        filtered_signal[i] = np.sum(weights * signal)

    return filtered_signal
def bivariate_filter_1d_optimized(signal, spatial_sigma, intensity_sigma):
    """
    Apply an optimized 1D bivariate (bilateral-like) filter to a 1D signal.

    Parameters:
        signal (np.ndarray): Input 1D signal.
        spatial_sigma (float): Spatial standard deviation.
        intensity_sigma (float): Intensity standard deviation.

    Returns:
        np.ndarray: Filtered 1D signal.
    """
    length = len(signal)
    filtered_signal = np.zeros_like(signal)

    # Precompute spatial weights
    spatial_weights = np.exp(-np.square(np.arange(-length, length)) / (2 * spatial_sigma ** 2))

    for i in range(length):
        # Define the window of interest
        start = max(0, i - length // 2)
        end = min(length, i + length // 2 + 1)

        # Slice the signal for the current window
        sub_signal = signal[start:end]

        # Compute intensity weights
        intensity_weights = np.exp(-np.square(sub_signal - signal[i]) / (2 * intensity_sigma ** 2))

        # Compute combined weights
        combined_weights = spatial_weights[length - (i - start):length + (end - i)] * intensity_weights

        # Normalize weights and filter
        filtered_signal[i] = np.sum(combined_weights * sub_signal) / np.sum(combined_weights)

    return filtered_signal


def bilateral_filter_1d(data, sigma_spatial, sigma_range, window_size):
    """
    Applies a 1D bilateral filter to the input data.

    Args:
      data: 1D input array.
      sigma_spatial: Standard deviation for spatial Gaussian kernel.
      sigma_range: Standard deviation for range Gaussian kernel.
      window_size: Size of the filtering window (odd integer).

    Returns:
      Filtered 1D array.
    """

    half_window = window_size // 2
    filtered_data = np.zeros_like(data)
    for i in range(half_window, len(data) - half_window):
        # Extract local neighborhood
        neighborhood = data[i - half_window: i + half_window + 1]
        # Calculate spatial weights
        spatial_weights = np.exp(-np.arange(-half_window, half_window + 1) ** 2 / (2 * sigma_spatial ** 2))
        # Calculate range weights
        intensity_diffs = neighborhood - data[i]
        range_weights = np.exp(-intensity_diffs ** 2 / (2 * sigma_range ** 2))
        # Combine weights
        weights = spatial_weights * range_weights.T
        # Normalize weights - Transpose back
        weights /= np.sum(weights)
        # Apply weighted average
        filtered_data[i] = np.sum(neighborhood * weights.T)
    return filtered_data
###########
# dateOfInterest = DT.datetime(2023, 11, 9, 13, 0, 0)  # "20231109T120000Z"
dateOfInterest = DT.datetime(2024, 7, 16) # repeats
# start getting imagery early so we can do the rest of the flow in parallel
# argusName = yellowfinLib.threadGetArgusImagery(dateOfInterest)
# yellowfinLib.plotPlanViewOnArgus(data, argusName, ofName='')
date_string = dateOfInterest.strftime('%Y%m%d')
yellowFinDatafname = f"/data/yellowfin/{dateOfInterest.year}/{date_string}/{date_string}_totalCombinedRawData.h5"
data = yellowfinLib.unpackYellowfinCombinedRaw(yellowFinDatafname)
# convert to all coords
coords = geoprocess.FRFcoord(p1=data['longitude'], p2=data['latitude'], coordType='LL')
pier_start = geoprocess.FRFcoord(0, 515, coordType='FRF')
pier_end = geoprocess.FRFcoord(534, 515, coordType='FRF')
line_numbers = sorted(np.unique(data['Profile_number']))[1:]
## isolate and focus on one FRF line profile
order = 2

sonar_bs = data['sonar_backscatter_out']
norm_sonar_bs = ((sonar_bs / sonar_bs.max()) * 255).astype('uint8')
s = cv2.bilateralFilter(norm_sonar_bs,  d=5, sigmaColor=10, sigmaSpace=10)

plt.figure()
plt.subplot(121)
plt.pcolormesh(sonar_bs)
plt.colorbar()
plt.xlim([0, 1000])
plt.subplot(122)
plt.pcolormesh(s)
plt.xlim([0, 1000])
plt.tight_layout()
plt.savefig(f'{date_string}_raw_backscatter')
plt.close()

sigma_spatial = 1000
sigma_range = 0.5
window_size = 100
for i in range(0, data['time'].shape[0], 100):
    filtered_data_instant = bilateral_filter_1d(data['sonar_backscatter_out'][i], sigma_spatial, sigma_range, window_size)
    filtered_data_inst_2 = bivariate_filter_1d_optimized(data['sonar_backscatter_out'][i],
                                                         spatial_sigma=sigma_spatial, intensity_sigma=1000)
    plt.figure()
    plt.plot(data['sonar_backscatter_out'][i])
    plt.plot(filtered_data_instant, label=f'filtered_{sigma_spatial}')
    plt.plot(filtered_data_inst_2, label=f'filtered_2_{sigma_spatial}')
    plt.legend()
    plt.savefig(f"{date_string}_{i:03}_instant_filtered_profile")
    plt.close()



cutoff = 1/10 #m
fig, axs = plt.subplots(ncols=1, nrows=len(line_numbers), figsize=(15, 8))
for i, lineNumber in enumerate(line_numbers):
    logic = (data['Profile_number'] == line_numbers[i])
    axs[i].plot(coords['xFRF'][logic], data['elevation'][logic], label='raw')
    axs[i].set_title(f'lineNumber {line_numbers[i]:.1f}')
    axs[i].set_ylabel('elevation[m]')
    for cutoff in [1/20, 1/50]:
        for order in [2, 5, 10]:
            b, a = signal.butter(order, cutoff, 'low', analog=False)
            output = signal.fi
            ltfilt(b, a, data['elevation'][logic])
            axs[i].plot(coords['xFRF'][logic], output, label=f'filtered c={cutoff} o={order}')

axs[i].legend()
plt.tight_layout()


# passedBathy = yellowfinLib.butter_lowpass_filter(data['elevation'], 5, fs=0.1, order=2)
#
# plt.figure(figsize=(12, 8));
# plt.subplot(211)
# plt.title('plan view of survey')
# plt.scatter(coords['xFRF'], coords['yFRF'], c=elevation_out[idxDataToSave],
#             vmax=-1)  # time_out[idxDataToSave])  #
# cbar = plt.colorbar()
# cbar.set_label('depth')
# plt.subplot(212)
# plt.title(f"profile at line y={np.median(coords['yFRF'][logic]).astype(int)}")
# plt.plot(coords['xFRF'][logic],
#          gnss_out[idxDataToSave][logic] - antenna_offset - sonar_instant_depth_out[idxDataToSave][logic],
#          label='instant depths')
# plt.plot(coords['xFRF'][logic],
#          gnss_out[idxDataToSave][logic] - antenna_offset - sonar_smooth_depth_out[idxDataToSave][logic],
#          label='smooth Depth')
# plt.plot(coords['xFRF'][logic], elevation_out[idxDataToSave][logic], label='chosen depths')
# plt.legend()
# plt.xlabel('xFRF')
# plt.ylabel('elevation NAVD88[m]')
# plt.tight_layout()
# plt.savefig(os.path.join(plotDir, 'singleProfile.png'))
