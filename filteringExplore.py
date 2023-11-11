from testbedutils imoprt geoprocess
from minio import Minio
import wget
import os

fname = '/data'

#https://pratiman-91.github.io/2020/06/30/Plotting-GeoTIFF-in-python.html

# convert to FRF
# if makePos is True and 'ncdu' in baseZipFiles:
coords = geoprocess.FRFcoord(lon_out[idxDataToSave], lat_out[idxDataToSave])
pierStart = geoprocess.FRFcoord(0, 515, coordType='FRF')
pierEnd = geoprocess.FRFcoord(534, 515, coordType='FRF')

## isolate and focus on one FRF line profile
minloc = 1270
maxloc = 1290
logic = (coords['yFRF'] > minloc) & (coords['yFRF'] < maxloc)

passedBathy = yellowfinLib.butter_lowpass_filter(elevation_out[idxDataToSave][logic], cutoff=1, fs=0.1, order=2)

plt.figure(figsize=(12, 8));
plt.subplot(211)
plt.title('plan view of survey')
plt.scatter(coords['xFRF'], coords['yFRF'], c=elevation_out[idxDataToSave], vmax=-1)  # time_out[idxDataToSave])  #
cbar = plt.colorbar()
cbar.set_label('depth')
plt.subplot(212)
plt.title(f"profile at line y={np.median(coords['yFRF'][logic]).astype(int)}")
plt.plot(coords['xFRF'][logic],
         gnss_out[idxDataToSave][logic] - antenna_offset - sonar_instant_depth_out[idxDataToSave][logic],
         label='instant depths')
plt.plot(coords['xFRF'][logic],
         gnss_out[idxDataToSave][logic] - antenna_offset - sonar_smooth_depth_out[idxDataToSave][logic],
         label='smooth Depth')
plt.plot(coords['xFRF'][logic], elevation_out[idxDataToSave][logic], label='chosen depths')
plt.legend()
plt.xlabel('xFRF')
plt.ylabel('elevation NAVD88[m]')
plt.tight_layout()
plt.savefig(os.path.join(plotDir, 'singleProfile.png'))


