import datetime as DT
import os
from matplotlib import pyplot as plt
import h5py
import rasterio
import wget
from rasterio import plot as rplt
from testbedutils import geoprocess


def getArgusImagery(dateOfInterest, ofName=None, imageType='timex', verbose=True):
    # client = Minio("coastalimaging.erdc.dren.mil")
    # ## now lets find what files are around
    # objects = client.list_objects('FrfTower', prefix="Processed/alignedObliques/c1", recursive=True,)
    baseURL = "https://coastalimaging.erdc.dren.mil/FrfTower/Processed/Orthophotos/cxgeo/"
    fldr = dateOfInterest.strftime("%Y_%m_%d")
    fname = f'{dateOfInterest.strftime("%Y%m%dT%H%M%SZ")}.FrfTower.cxgeo.{imageType}.tif'
    wgetURL = os.path.join(baseURL, fldr, fname)
    if ofName is None:
        ofName = os.path.join(os.getcwd(), os.path.basename(wgetURL))
    wget.download(wgetURL, ofName)
    if verbose: print(f"retrieved {ofName}")
    return ofName

def unpackYellowfinCombinedRaw(fname):
    data = {}
    with h5py.File(fname, 'r') as hf:
        for var in ['time', 'longitude', 'latitude', 'elevation', 'fix_quality_GNSS', 'sonar_smooth_depth',
                    'sonar_smooth_confidence', 'sonar_instant_depth', 'sonar_instant_depth', 'sonar_instant_confidence',
                    'sonar_backscatter_out', 'bad_lat', 'bad_lon', 'xFRF', 'yFRF', 'Profile_number']:
            data[var] = hf.get(var)[:]
    return data

def plotPlanViewOnArgus(data, geoTifName):
    """plots a survey path over a geotiff at the FRF (assumes NC stateplane)
    Args:
        data: this is a dictionary of data loaded with keys of 'longitude', 'latitude', 'elevation'
        geoTifName: this is a filenamepath of a geotiff file over which elevation and path data are to be overlayed

    """
    coords = geoprocess.FRFcoord(data['longitude'], data['latitude'])s
    timex = rasterio.open(geoTifName)
    # array = timex.read()  # for reference, this pulls the image data out of the geotiff object
    ## now make plot
    plt.figure(figsize=(14,10))
    ax1 = plt.subplot()
    aa = rplt.show(timex, ax=ax1)
    a = ax1.scatter(coords['StateplaneE'], coords['StateplaneN'], c=data['elevation'], vmin=-8)
    cbar = plt.colorbar(a)
    cbar.set_label('depths')
    ax1.set_xlabel('NC stateplane Easting')
    ax1.set_ylabel('NC stateplane Northing')
    plt.savefig('Overview_on_Argus.png')
    plt.close()

yellowFinDatafname = '/data/yellowfin/20231109/20231109_totalCombinedRawData.h5'
data = unpackYellowfinCombinedRaw(yellowFinDatafname)
# convert to all coords
pierStart = geoprocess.FRFcoord(0, 515, coordType='FRF')
pierEnd = geoprocess.FRFcoord(534, 515, coordType='FRF')

#now get imagery
dateOfInterest = DT.datetime(2023, 11, 9, 13, 0, 0)  # "20231109T120000Z"
imageType = 'timex'
# a = getArgusImagery(dateOfInterest)
geoTifName = "/home/spike/repos/SWACSS/20231109T130000Z.FrfTower.cxgeo.timex.tif"
# https://pratiman-91.github.io/2020/06/30/Plotting-GeoTIFF-in-python.html
plotPlanViewOnArgus(data, geoTifName)

## isolate and focus on one FRF line profile
minloc = 1270
maxloc = 1290
logic = (coords['yFRF'] > minloc) & (coords['yFRF'] < maxloc)

passedBathy = yellowfinLib.butter_lowpass_filter(elevation_out[idxDataToSave][logic], cutoff=1, fs=0.1, order=2)

plt.figure(figsize=(12, 8));
plt.subplot(211)
plt.title('plan view of survey')
plt.scatter(coords['xFRF'], coords['yFRF'], c=elevation_out[idxDataToSave],
            vmax=-1)  # time_out[idxDataToSave])  #
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
