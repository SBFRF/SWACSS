import logging
import os
import sys
import matplotlib
from scipy import interpolate, signal

import py2netCDF

matplotlib.use('TkAgg')
import yellowfinLib
import datetime as DT
from matplotlib import pyplot as plt
import numpy as np
import h5py
import pandas as pd
import glob
import zipfile
import tqdm
from testbedutils import geoprocess
import argparse, logging

__version__ = 0.2
def parse_args(__version__):
    parser = argparse.ArgumentParser(f"PPK processing for yellowfin (V{__version__})", add_help=True)
    # datadir, geoid, makePos = True, verbose = 1
    # Command-Line Interface: (REQUIRED) Flags
    parser.add_argument('-d', '--data_dir', type=str, metavar=True, help="directory to process",
                        required=True)
    # parser.add_argument('-t', '--target', type=str, nargs=2, metavar=("path", "label"), required=True,
    #                     action='append', help="file path to target ground truth")

    # Command-Line Interface: (OPTIONAL) Flags
    parser.add_argument('-g', '--geoid_file', type=str, default='ref/g2021bu0.bin', metavar='',
                        help="binary geoid file")
    parser.add_argument('-p', '--make_pos', type=bool, default=True,
                        help="make posfile (True) or provide one through external environment (false)")
    parser.add_argument('-v', '--verbosity', type=int, default=2, metavar='',
                        help='sets verbosity for debug, 1=Debug (most), 2=Info (normal), 3=Warning (least)')
    return parser.parse_args()

def verbosity_conversion(verbose: int):
    if verbose == 1:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 2:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 3:
        logging.basicConfig(level=logging.WARN)
    else:
        raise EnvironmentError('logging verbosity is wrong!')
def main(datadir, geoid, makePos=True, verbose=2):

    verbosity_conversion(verbose)
    antenna_offset = 0.25  # meters between the antenna phase center and sounder head
    PPKqualityThreshold = 1
    smoothedSonarConfidence = 60
    instantSonarConfidence = 99
    UTCthresh = DT.datetime(2023, 7,
                            10)  # date that Pi computer was changed to UTC time (will adjust timezone manually before this date)
    sonarMethod = 'instant'
    RTKLibexecutablePath = 'ref/rnx2rtkp'
    ####################################################################################################################

    if datadir.endswith('/'): datadir = datadir[:-1]
    ## Define all paths for the workflow
    timeString = os.path.basename(datadir)
    plotDir = os.path.join(datadir, 'figures')
    os.makedirs(plotDir, exist_ok=True)  # make folder structure if its not already made
    argusGeotiff = yellowfinLib.threadGetArgusImagery(DT.datetime.strptime(timeString, '%Y%m%d') +
                                                      DT.timedelta(hours=14),
                                                      ofName=os.path.join(plotDir, f'Argus_{timeString}'),)

    # sonar data
    fpathSonar = os.path.join(datadir, 's500')  # reads sonar from here
    saveFnameSonar = os.path.join(datadir, f'{timeString}_sonarRaw.h5')  # saves sonar file here

    # NMEA data from sonar, this is not Post Processed Kinematic (PPK) data.  It is used for only cursory or
    # introductory look at the data
    fpathGNSS = os.path.join(datadir, 'nmeadata')  # load NMEA data from this location
    saveFnameGNSS = os.path.join(datadir, f'{timeString}_gnssRaw.h5')  # save nmea data to this location

    # RINEX data
    # look for all subfolders with RINEX in the folder name inside the "datadir" emlid ppk processor
    fpathEmlid = os.path.join(datadir, 'emlidRaw')
    saveFnamePPK = os.path.join(datadir, f'{timeString}_ppkRaw.h5')


    logging.debug(f"saving intermediate files for sonar here: {saveFnameSonar}")
    logging.debug(f"saving intermediate files for sonar here: {saveFnamePPK}")
    logging.debug(f"saving intermediate files for GNSS here: {saveFnameGNSS}")
    ## load files
    # yellowfinLib.loadSonar_s500_binary(fpathSonar, outfname=saveFnameSonar, verbose=verbose)

    # then load NMEA files
    # yellowfinLib.load_yellowfin_NMEA_files(fpathGNSS, saveFname=saveFnameGNSS,
    #                                        plotfname=os.path.join(plotDir, 'GPSpath_fromNMEAfiles.png'),
    #                                        verbose=verbose)
    if makePos == True:
        # find folders with raw rinex
        rinexZipFiles = glob.glob(os.path.join(fpathEmlid, '*RINEX*.zip'))
        try:  # we've got a zip file from CORS station
            baseZipFiles = glob.glob(os.path.join(datadir, 'CORS', '*.zip'))[0]
            with zipfile.ZipFile(baseZipFiles, 'r') as zip_ref:
                zip_ref.extractall(path=baseZipFiles[:-4])
            baseFname = glob.glob(os.path.join(os.path.splitext(baseZipFiles)[0], '*o'))[0]
            navFile = glob.glob(os.path.join(os.path.splitext(baseZipFiles)[0], '*n'))[0]
            sp3fname = glob.glob(os.path.join(os.path.splitext(baseZipFiles)[0], '*sp3'))
            if len(sp3fname) > 0:
                sp3fname = sp3fname[0]
            else:
                sp3fname = ''
        except IndexError:  # we downloaded the observation files
            raise NotImplementedError('Need to develope methods to use the raw observations not from CORS')
        # unzip all the rinex Files
        for ff in rinexZipFiles:
            with zipfile.ZipFile(ff, 'r') as zip_ref:
                zip_ref.extractall(path=ff[:-4])
            # identify and process rinex to Pos files
            flist_rinex = glob.glob(ff[:-4] + "/*")
            roverFname = flist_rinex[np.argwhere([i.endswith('O') for i in flist_rinex]).squeeze()]
            outfname = os.path.join(os.path.dirname(roverFname), os.path.basename(flist_rinex[0])[:-3] + "pos")
            # use below if the rover nav file is the right call
            yellowfinLib.makePOSfileFromRINEX(roverObservables=roverFname, baseObservables=baseFname, navFile=navFile,
                                              outfname=outfname, executablePath=RTKLibexecutablePath, sp3=sp3fname)

    # Now find all the folders that have ppk data in them (*.pos files in folders that have "raw" in them)
    # now identify the folders that have rinex in them
    fldrlistPPK = []  # initalize list for appending RINEX folder in
    [fldrlistPPK.append(os.path.join(fpathEmlid, fname)) for fname in os.listdir(fpathEmlid) if
     'raw' in fname and '.zip' not in fname]

    logging.info('load PPK pos files ---- THESE ARE WGS84!!!!!!!!!!!!!!')
    try:
        T_ppk = yellowfinLib.loadPPKdata(fldrlistPPK)
        T_ppk.to_hdf(saveFnamePPK, 'ppk')  # now save the h5 intermediate file
    except KeyError:
        raise FileExistsError("the pos file hasn't been loaded, manually produce or turn on RTKlib processing")
    # 1. time in seconds to adjust to UTC from ET (varies depending on time of year!!!)
    if (T_ppk['datetime'].iloc[0].replace(tzinfo=None) < UTCthresh) & (
            int(T_ppk['datetime'].iloc[0].day_of_year) > 71) & (int(T_ppk['datetime'].iloc[0].day_of_year) < 309):
        ET2UTC = 5 * 60 * 60
    elif (T_ppk['datetime'].iloc[0].replace(tzinfo=None) < UTCthresh) & (
            int(T_ppk['datetime'].iloc[0].day_of_year) < 71) & (int(T_ppk['datetime'].iloc[0].day_of_year) > 309):
        ET2UTC = 4 * 60 * 60
    else:
        ET2UTC = 0  # time's already in UTC
    if ET2UTC > 0:
        print(" I'm using a 'dumb' conversion from ET to UTC")

    # 6.2: load all files we created in previous steps
    sonarData = yellowfinLib.load_h5_to_dictionary(saveFnameSonar)
    payloadGpsData = yellowfinLib.load_h5_to_dictionary(saveFnameGNSS)  # this is used for the pc time adjustement
    T_ppk = pd.read_hdf(saveFnamePPK)

    # Adjust GNSS time by the Leap Seconds https://www.cnmoc.usff.navy.mil/Our-Commands/United-States-Naval-Observatory/Precise-Time-Department/Global-Positioning-System/USNO-GPS-Time-Transfer/Leap-Seconds/
    # T_ppk['epochTime'] = T_ppk['epochTime'] - 18  # 18 is leap second adjustment
    # T_ppk['datetime'] = T_ppk['datetime'] - DT.timedelta(seconds=18)  # making sure both are equal

    # convert raw ellipsoid values from satellite measurement to that of a vertical datum.  This uses NAVD88 [m] NAD83
    T_ppk['GNSS_elevation_NAVD88'] = yellowfinLib.convertEllipsoid2NAVD88(T_ppk['lat'], T_ppk['lon'], T_ppk['height'],
                                                                          geoidFile=geoid)
    # 6.3: now plot my time offset between GPS and sonar
    pc_time_off = payloadGpsData['pc_time_gga'] + ET2UTC - payloadGpsData['gps_time']

    # Compare GPS data to make sure timing is ok
    plt.figure()
    plt.plot(pc_time_off, '.')
    plt.title('time offset between pc time and GPS time')
    plt.xlabel('PC time')
    plt.ylabel('PC time - GGA string time (+leap seconds)')
    plt.savefig(os.path.join(plotDir, 'clockOffset.png'))
    print(f'the PC time (sonar time stamp) is {np.median(pc_time_off):.2f} seconds behind the GNSS timestamp')

    # 6.4 Use the cerulean instantaneous bed detection since not sure about delay with smoothed
    # adjust time of the sonar time stamp with timezone shift (ET -> UTC) and the timeshift between the computer and GPS
    sonarData['time'] = sonarData['time'] + ET2UTC - np.median(pc_time_off)  # convert to UTC
    if sonarMethod == 'smooth':
        sonar_range = sonarData['smooth_depth_m']
        qualityLogic = sonarData['smoothed_depth_measurement_confidence'] > smoothedSonarConfidence
    elif sonarMethod == 'instant':
        sonar_range = sonarData['this_ping_depth_m']
        qualityLogic = sonarData['this_ping_depth_measurement_confidence'] > instantSonarConfidence
    else:
        raise ValueError('acceptable sonar methods include ["instant", "smooth"]')
    # use the above to adjust whether you want smoothed/filtered data or raw ping depth values

    # 6.5 now plot sonar with time
    plt.figure(figsize=(18, 6))
    cm = plt.pcolormesh([DT.datetime.utcfromtimestamp(i) for i in sonarData['time']], sonarData['range_m'],
                        sonarData['profile_data'])
    cbar = plt.colorbar(cm)
    cbar.set_label('backscatter')
    plt.plot([DT.datetime.utcfromtimestamp(i) for i in sonarData['time']], sonarData['this_ping_depth_m'], 'r-', lw=1,
             label='this ping Depth')
    plt.plot([DT.datetime.utcfromtimestamp(i) for i in sonarData['time']], sonarData['smooth_depth_m'], 'k-', lw=2,
             label='smooth Depth')
    plt.ylim([10, 0])
    plt.legend(loc='lower left')
    # plt.gca().invert_yaxis()
    plt.tight_layout(rect=[0.05, 0.05, 0.99, 0.99], w_pad=0.01, h_pad=0.01)
    plt.savefig(os.path.join(plotDir, 'SonarBackScatter.png'))

    # 6.6 Now lets take a look at all of our data from the different sources
    plt.figure(figsize=(10, 4))
    plt.suptitle('all data sources elevation', fontsize=20)
    plt.title('These data need to overlap in time for processing to work')
    plt.plot([DT.datetime.utcfromtimestamp(i) for i in sonarData['time']], sonar_range, 'b.', label='sonar depth')
    plt.plot([DT.datetime.utcfromtimestamp(i) for i in payloadGpsData['gps_time']], payloadGpsData['altMSL'], '.g',
             label='L1 (only) GPS elev (MSL)')
    plt.plot([DT.datetime.utcfromtimestamp(i) for i in T_ppk['epochTime']], T_ppk['GNSS_elevation_NAVD88'], '.r',
             label='ppk elevation [NAVD88 m]')
    plt.ylim([0, 10])
    plt.ylabel('elevation [m]')
    plt.xlabel('epoch time (s)')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(plotDir, 'AllData.png'))

    # 6.7 # plot sonar, select indices of interest, and then second subplot is time of interest
    plt.figure(figsize=(10, 4))
    plt.subplot(211)
    plt.title('all data, select start/end point for measured depths')
    plt.plot(sonar_range)
    plt.ylim([0, 10])
    d = plt.ginput(2, timeout=-999)
    plt.subplot(212)
    # Now pull corresponding indices for sonar data for same time
    assert len(d) == 2, "need 2 points from mouse clicks"
    sonarIndicies = np.arange(np.floor(d[0][0]).astype(int), np.ceil(d[1][0]).astype(int))
    plt.plot(sonar_range[sonarIndicies])
    plt.title('my selected data to proceed with')
    plt.tight_layout()

    plt.savefig(os.path.join(plotDir, 'subsetForCrossCorrelation.png'))

    # now identify corresponding times from ppk GPS to those times of sonar that we're interested in
    indsPPK = np.where((T_ppk['epochTime'] >= sonarData['time'][sonarIndicies[0]]) & (
            T_ppk['epochTime'] <= sonarData['time'][sonarIndicies[-1]]))[0]

    # 6.7 interpolate and calculate the phase offset between the signals

    ## now interpolate the lower sampled (sonar 3.33 hz) to the higher sampled data (gps 10 hz)
    # identify common timestamp to interpolate to at higher frequency
    commonTime = np.linspace(T_ppk['epochTime'][indsPPK[0]], T_ppk['epochTime'][indsPPK[-1]],
                             int((T_ppk['epochTime'][indsPPK[-1]] - T_ppk['epochTime'][indsPPK[0]]) / .1),
                             endpoint=True)

    # always use instant ping for time offset calculation
    f = interpolate.interp1d(sonarData['time'], sonarData['this_ping_depth_m'])
    sonar_range_i = f(commonTime)
    f = interpolate.interp1d(T_ppk['epochTime'], T_ppk['height'])
    ppkHeight_i = f(commonTime)
    # now i have both signals at the same time stamps
    phaseLagInSamps, phaseLaginTime = yellowfinLib.findTimeShiftCrossCorr(signal.detrend(ppkHeight_i),
                                                                          signal.detrend(sonar_range_i),
                                                                          sampleFreq=np.median(np.diff(commonTime)))

    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(311)
    plt.plot(T_ppk['epochTime'][indsPPK], T_ppk['GNSS_elevation_NAVD88'][indsPPK], label='ppk elevation NAVD88 m')
    plt.plot(sonarData['time'][sonarIndicies], sonar_range[sonarIndicies], label='sonar_raw')
    plt.legend()

    plt.subplot(312, sharex=ax1)
    plt.title(f"sonar data needs to be adjusted by {phaseLaginTime} seconds")
    plt.plot(commonTime, signal.detrend(ppkHeight_i), label='ppk input')
    plt.plot(commonTime, signal.detrend(sonar_range_i), label='sonar input')
    plt.plot(commonTime + phaseLaginTime, signal.detrend(sonar_range_i), '.', label='interp _sonar shifted')
    plt.legend()

    plt.subplot(313, sharex=ax1)
    plt.title('shifted residual between sonar and GNSS (should be 0)')
    plt.plot(commonTime + phaseLaginTime, signal.detrend(sonar_range_i) - signal.detrend(ppkHeight_i), '.',
             label='residual')
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plotDir, 'subsetAfterCrossCorrelation.png'))

    print(f"sonar data adjusted by {phaseLaginTime:.3f} seconds")

    ## now process all data for saving to file
    sonar_time_out = sonarData['time'] + phaseLaginTime

    ## ok now put the sonar data on the GNSS timestamps which are decimal seconds.  We can do this with sonar_time_out,
    # because we just adjusted by the phase lag to make sure they are time synced.
    timeOutInterpStart = np.ceil(sonar_time_out.min() * 10) / 10  # round to nearest 0.1
    timeOutInterpEnd = np.floor(sonar_time_out.max() * 10) / 10  # round to nearest 0.1
    # create a timestamp for data to be output and in phase with that of the ppk gps data which are on the 0.1 s
    time_out = np.linspace(timeOutInterpStart, timeOutInterpEnd, int((timeOutInterpEnd - timeOutInterpStart) / 0.1),
                           endpoint=True)

    print("here's where some better filtering could be done, probably worth saving an intermediate product here "
          "for future revisit")

    print(f"for now we're only saving/logging values that have a GNSS fix quality of {PPKqualityThreshold} and a "
          f"sonar confidence > {smoothedSonarConfidence}")

    # now put relevant GNSS and sonar on output timestamps
    # initalize out variables
    sonar_smooth_depth_out, sonar_smooth_confidence_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan
    sonar_instant_depth_out, sonar_instant_confidence_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan
    sonar_backscatter_out = np.zeros((time_out.shape[0], sonarData['range_m'].shape[0])) * np.nan
    bad_lat_out, bad_lon_out, lat_out, lon_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan, np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan
    elevation_out, fix_quality = np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan
    gnss_out, sonar_out = np.zeros_like(bad_lat_out) * np.nan, np.zeros_like(time_out) * np.nan
    # loop through my common time (.1 s increment) and find associated sonar and gnss values; this might be slow
    for tidx, tt in tqdm.tqdm(enumerate(time_out)):
        idxTimeMatchGNSS, idxTimeMatchGNSS = None, None

        # first find if theres a time match for sonar
        sonarlogic = np.abs(np.ceil(tt * 10) / 10 - np.ceil(sonar_time_out * 10) / 10)
        if sonarlogic.min() <= 0.2:  # 0.2  with a sampling of <0-2, it should identify the nearest sample (at 0.3s interval)
            idxTimeMatchSonar = np.argmin(sonarlogic)
        # then find comparable time match for ppk
        ppklogic = np.abs(np.ceil(tt * 10) / 10 - np.ceil(T_ppk['epochTime'].array * 10) / 10)
        if ppklogic.min() <= 0.101:  # .101 handles numerics
            idxTimeMatchGNSS = np.argmin(ppklogic)

        # if we have both, then we log the data
        if idxTimeMatchGNSS is not None and idxTimeMatchSonar is not None:  # we have matching data
            # now apply it
            if T_ppk['Q'][idxTimeMatchGNSS] <= PPKqualityThreshold and qualityLogic[idxTimeMatchSonar]:
                # log matching data that meets quality metrics
                sonar_smooth_depth_out[tidx] = sonarData['smooth_depth_m'][idxTimeMatchSonar]
                sonar_instant_depth_out[tidx] = sonarData['this_ping_depth_m'][idxTimeMatchSonar]
                sonar_smooth_confidence_out[tidx] = sonarData['smoothed_depth_measurement_confidence'][
                    idxTimeMatchSonar]
                sonar_instant_confidence_out[tidx] = sonarData['this_ping_depth_measurement_confidence'][
                    idxTimeMatchSonar]
                sonar_backscatter_out[tidx] = sonarData['profile_data'][:, idxTimeMatchSonar]
                lat_out[tidx] = T_ppk['lat'][idxTimeMatchGNSS]
                lon_out[tidx] = T_ppk['lon'][idxTimeMatchGNSS]
                gnss_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS]
                fix_quality[tidx] = T_ppk['Q'][idxTimeMatchGNSS]
                # now log elevation outs depending on which sonar i want to log
                if sonarMethod == 'smooth':
                    elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                          sonarData['smooth_depth_m'][idxTimeMatchSonar]
                    sonar_out[tidx] = sonarData['smooth_depth_m'][idxTimeMatchSonar]

                elif sonarMethod == 'instant':
                    elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                          sonarData['this_ping_depth_m'][idxTimeMatchSonar]
                    sonar_out[tidx] = sonarData['this_ping_depth_m'][idxTimeMatchSonar]
                else:
                    raise ValueError('acceptable sonar methods include ["instant", "smooth"]')

            # now log bad locations for quality plotting
            if T_ppk['Q'][idxTimeMatchGNSS] <= PPKqualityThreshold and not qualityLogic[idxTimeMatchSonar]:
                bad_lat_out[tidx] = T_ppk['lat'][idxTimeMatchGNSS]
                bad_lon_out[tidx] = T_ppk['lon'][idxTimeMatchGNSS]
    # identify data that are not nan's to save
    idxDataToSave = np.argwhere(~np.isnan(sonar_smooth_depth_out)).squeeze()  # identify data that are not NaNs

    fs = 16
    # make a final plot of all the processed data
    pierStart = geoprocess.FRFcoord(0, 515, coordType='FRF')
    pierEnd = geoprocess.FRFcoord(534, 515, coordType='FRF')

    plt.figure(figsize=(12, 8))
    plt.scatter(lon_out[idxDataToSave], lat_out[idxDataToSave], c=elevation_out[idxDataToSave], vmax=-0.5,
                label='processed depths')
    cbar = plt.colorbar()
    cbar.set_label('depths NAVD88 [m]', fontsize=fs)
    plt.plot(T_ppk['lon'], T_ppk['lat'], 'k.', ms=0.25, label='vehicle trajectory')
    plt.plot(bad_lon_out, bad_lat_out, 'rx', ms=3, label='bad sonar data, good GPS')
    plt.plot([pierStart['Lon'], pierEnd['Lon']], [pierStart['Lat'], pierEnd['Lat']], 'k-', lw=5, label='FRF pier')
    plt.ylabel('latitude', fontsize=fs)
    plt.xlabel('longitude', fontsize=fs)
    plt.title(f'final data with elevations {timeString}', fontsize=fs + 4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plotDir, 'FinalDataProduct.png'))


    FRF = True
    try:
        coords = geoprocess.FRFcoord(lon_out[idxDataToSave], lat_out[idxDataToSave])

        minloc = 800
        maxloc = 1000
        logic = (coords['yFRF'] > minloc) & (coords['yFRF'] < maxloc)

        plt.figure(figsize=(12, 8));
        plt.subplot(211)
        plt.title('plan view of survey')
        plt.scatter(coords['xFRF'], coords['yFRF'], c=elevation_out[idxDataToSave], vmax=-1)
        cbar = plt.colorbar()
        cbar.set_label('depth')
        plt.subplot(212)
        plt.title(f"profile at line y={np.median(coords['yFRF'][logic]).astype(int)}")
        plt.plot(coords['xFRF'][logic],
                 gnss_out[idxDataToSave][logic] - antenna_offset - sonar_instant_depth_out[idxDataToSave][logic], '.',
                 label='instant depths')
        plt.plot(coords['xFRF'][logic],
                 gnss_out[idxDataToSave][logic] - antenna_offset - sonar_smooth_depth_out[idxDataToSave][logic], '.',
                 label='smooth Depth')
        plt.plot(coords['xFRF'][logic], elevation_out[idxDataToSave][logic], '.', label='chosen depths')
        plt.legend()
        plt.xlabel('xFRF')
        plt.ylabel('elevation NAVD88[m]')
        plt.tight_layout()
        plt.savefig(os.path.join(plotDir, 'singleProfile.png'))

        data = {'time': time_out[idxDataToSave], 'date': DT.datetime.strptime(timeString, "%Y%m%d").timestamp(),
                'Latitude': lat_out[idxDataToSave], 'Longitude': lon_out[idxDataToSave],
                'Northing': coords['StateplaneN'], 'Easting': coords['StateplaneE'], 'xFRF': coords['xFRF'],
                'yFRF': coords['yFRF'], 'Elevation': elevation_out[idxDataToSave],
                'Profile_number': np.ones_like(elevation_out[idxDataToSave]) * -999,
                'Survey_number': np.ones_like(elevation_out[idxDataToSave]) * -999,
                'Ellipsoid': np.ones_like(elevation_out[idxDataToSave]) * -999}

        data['UNIX_timestamp'] = data['time']
        data = yellowfinLib.transectSelection(pd.DataFrame.from_dict(data), outputDir=plotDir)
        ## now make netCDF files
        ofname = os.path.join(datadir, f'FRF_geomorphology_elevationTransects_survey_{timeString}.nc')
        data['Profile_number'] = data.pop('profileNumber')
        # data['Profile_number'].iloc[np.isnan(data['Profile_number'])] = -999
        data['Profile_number'].iloc[np.argwhere(data['Profile_number'].isnull()).squeeze()] = -999
        py2netCDF.makenc_generic(ofname, globalYaml='yamlFile/transect_global.yml',
                                 varYaml='yamlFile/transect_variables.yml', data=data)
        yellowfinLib.plotPlanViewOnArgus(data, argusGeotiff, ofName=os.path.join(plotDir, 'yellowfinDepthsOnArgus.png'))

    except:
        FRF = False

    outputfile = os.path.join(datadir, f'{timeString}_totalCombinedRawData.h5')
    with h5py.File(outputfile, 'w') as hf:
        hf.create_dataset('time', data=time_out[idxDataToSave])
        hf.create_dataset('longitude', data=lon_out[idxDataToSave])
        hf.create_dataset('latitude', data=lat_out[idxDataToSave])
        hf.create_dataset('elevation', data=elevation_out[idxDataToSave])
        hf.create_dataset('fix_quality_GNSS', data=fix_quality[idxDataToSave])
        hf.create_dataset('sonar_smooth_depth', data=sonar_smooth_depth_out[idxDataToSave])
        hf.create_dataset('sonar_smooth_confidence', data=sonar_smooth_confidence_out[idxDataToSave])
        hf.create_dataset('sonar_instant_depth', data=sonar_instant_depth_out[idxDataToSave])
        hf.create_dataset('sonar_instant_confidence', data=sonar_instant_confidence_out[idxDataToSave])
        hf.create_dataset('sonar_backscatter_out', data=sonar_backscatter_out[idxDataToSave])
        hf.create_dataset('bad_lat', data=bad_lat_out)
        hf.create_dataset('bad_lon', data=bad_lon_out)
        if FRF is not False:
            hf.create_dataset('xFRF', data=coords['xFRF'])
            hf.create_dataset('yFRF', data=coords['yFRF'])
        hf.create_dataset('Profile_number', data=data['Profile_number'])


if __name__ == "__main__":
    # filepath = '/data/yellowfin/20231109'  # 327'  # 04' #623' #705'
    args = parse_args(__version__)
    assert os.path.isdir(args.data_dir), "check your input filepath, code doesn't see the folder"

    main(args.data_dir, geoid=args.geoid_file, makePos=args.make_pos, verbose=args.verbosity)
    print(f"success processing {args.data_dir}")

