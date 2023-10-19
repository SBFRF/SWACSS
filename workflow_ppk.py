import os
import matplotlib
matplotlib.use('TkAgg')
import yellowfinLib
import datetime as DT
from matplotlib import pyplot as plt
import numpy as np
import h5py
import pandas as pd

def main(timeString, database, geoid):
    UTCthresh = DT.datetime(2023, 7, 10)  # date that Pi computer was changed to UTC time

    ## Define all paths for the workflow
    datadir = os.path.join(database, timeString)
    plotDir = os.path.join(datadir, 'figures')
    os.makedirs(plotDir, exist_ok=True)  # make folder structure if its not already made

    # sonar data
    fpathSonar = os.path.join(datadir, 's500')  # reads sonar from here
    saveFnameSonar = os.path.join(datadir, f'{timeString}_sonarRaw.h5')  # saves sonar file here

    # NMEA data from sonar, this is not Post Processed Kinematic (PPK) data.  It is used for only cursory or
    # introductory look at the data
    fpathGNSS = os.path.join(datadir, 'nmeaData')  # load NMEA data from this location
    saveFnameGNSS = os.path.join(datadir, f'{timeString}_gnssRaw.h5')  # save nmea data to this location

    # RINEX data
    # look for all subfolders with RINEX in the folder name inside the "datadir" emlid ppk processor
    saveFnamePPK = os.path.join(datadir, f'{timeString}_ppkRaw.h5')

    ## load files
    yellowfinLib.loadSonar_s500_binary(fpathSonar, outfname=saveFnameSonar, verbose=False)

    # then load NMEA files
    yellowfinLib.load_yellowfin_NMEA_files(fpathGNSS, saveFname=saveFnameGNSS,
                                           plotfname=os.path.join(datadir, 'figures', 'GPSpath.png'))

    # first find all the folders that have ppk data in them (*.pos files in folders that have "raw" in them)
    fldrlistPPK = [] # initalize list for appending RINEX folder in
    [fldrlistPPK.append(os.path.join(datadir, fname)) for fname in os.listdir(datadir) if 'raw' in fname and '.zip' not in fname]
    T_ppk = yellowfinLib.loadPPKdata(fldrlistPPK)
    T_ppk.to_hdf(saveFnamePPK, 'ppk') # now save the h5 intermediate file

    # 1. time in seconds to adjust to UTC from ET (varies depending on time of year!!!)
    if (T_ppk['datetime'].iloc[0] < UTCthresh & int(T_ppk['datetime'].iloc[0].day_of_year) > 71 &
            int(T_ppk['datetime'].iloc[0].day_of_year) < 309):
        ET2UTC = 5 * 60 * 60
    elif (T_ppk['datetime'].iloc[0] < UTCthresh & int(T_ppk['datetime'].iloc[0].day_of_year) < 71 &
          int(T_ppk['datetime'].iloc[0].day_of_year) > 309):
        ET2UTC = 4 * 60 * 60
    else:
        ET2UTC = 0 # time's already in UTC
    if ET2UTC > 0: print(" I'm using a 'dumb' conversion from ET to UTC")

    # 6.2
    # load all files we created in previous steps
    S1 = yellowfinLib.load_h5_to_dictionary(saveFnameSonar)
    GPS = yellowfinLib.load_h5_to_dictionary(saveFnameGNSS)  # this is used for the pc time adjustement
    T_ppk = pd.read_hdf(saveFnamePPK)

    # Adjust GNSS time by the Leap Seconds
    T_ppk['epochTime'] = T_ppk['epochTime'] - 18  # 18 is leap second adjustment
    T_ppk['datetime'] = T_ppk['datetime'] - DT.timedelta(seconds=18)  # making sure both are equal

    # convert raw elipsoid values from satellite measurement to that of a vertical datum.  This uses NAVD88 [m] NAD83
    T_ppk['GNSS_elevation_NAVD88'] = yellowfinLib.convertEllipsoid2NAVD88(T_ppk['lat'], T_ppk['lon'], T_ppk['height'],
                                                             geoidFile=geoidFileLoc)

    # 6.3
    # now plot my time offset between GPS and sonar
    pc_time_off = GPS['pc_time_gga'] + ET2UTC - GPS['gps_time']

    # Compare GPS data to make sure timing is ok
    plt.figure()
    plt.plot(GPS['gps_time'], pc_time_off, '.')
    plt.title('time offset between pc time and GPS time')
    plt.xlabel('gps time')
    plt.ylabel('time offset')
    plt.show()
    plt.savefig(os.path.join(plotDir, 'clockOffset.png'))
    print(f'the sonar is {np.median(pc_time_off):.2f} seconds behind the GNSS timestamp')

    # 6.4 Use the cerulean instantaneous bed detection since not sure about delay with smoothed
    # adjust time of the sonar time stamp with timezone shift (ET -> UTC) and the timeshift between the computer and GPS
    S1['time'] = S1['time'] + ET2UTC - np.median(pc_time_off)  # DT.timedelta(hours=5)  # convert to UTC
    sonar_range = S1['smooth_depth_m']  # S1['this_ping_depth_m']
    # use the above to adjust whether you want smoothed/filtered data or raw ping depth values

    # 6.5 now plot sonar with time
    plt.figure(figsize=(18,6))
    cm = plt.pcolormesh(S1['time'], S1['range_m'], S1['profile_data'])
    cbar = plt.colorbar(cm)
    cbar.set_label('backscatter')
    plt.plot(S1['time'], S1['smooth_depth_m'], 'k-', lw=2, label='smooth Depth')
    plt.plot(S1['time'], S1['this_ping_depth_m'], 'r-', lw=1, label='this ping Depth')
    plt.ylim([10, 0])
    plt.legend(loc='lower left')
    #plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plotDir, 'SonarBackScatter.png'))

    # 6.6 Now lets take a look at all of our data from the different sources
    plt.figure(figsize=(10, 4))
    plt.title('all data sources elevation', fontsize=20)
    plt.plot(S1['time'], sonar_range, 'b.', label='sonar instant depth')
    plt.plot(GPS['gps_time'], GPS['altMSL'], '.g', label='L1 (only) GPS elev (MSL)')
    plt.plot(T_ppk['epochTime'], T_ppk['GNSS_elevation_NAVD88'], '.r', label='ppk elevation [NAVD88 m]')
    plt.ylim([0, 20])
    plt.ylabel('elevation [m]')
    plt.xlabel('epoch time (s)')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(plotDir, 'AllData.png'))
    print('These data need to overlap!')

    # 6.7 # plot sonar, select indices of interest, and then second subplot is time of interest
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.title('all data, select start/end point for measured depths\nadd extra time for PPK offset')
    plt.plot(sonar_range)
    # plt.ylim([0,10])
    d = plt.ginput(2)

    plt.subplot(212)
    # Now pull corresponding indices for sonar data for same time
    sonarIndicies = np.arange(np.floor(d[0][0]).astype(int), np.ceil(d[1][0]).astype(int))
    plt.plot(sonar_range[sonarIndicies])
    plt.title('my selected data to proceed with')
    plt.tight_layout()
    # plt.ylim([0,10])
    plt.savefig(os.path.join(plotDir, 'subsetForCrossCorrelation.png'))


    # now identify corresponding times from ppk GPS to those times of sonar that we're interested in
    indsPPK = np.where((T_ppk['epochTime'] >= S1['time'][sonarIndicies[0]]) &
                       (T_ppk['epochTime'] <= S1['time'][sonarIndicies[-1]]))[0]

    # 6.7 interpolate and calculate the phase offset between the signals
    from scipy import interpolate, signal

    ## now interpolate the lower sampled (sonar 3.33 hz) to the higher sampled data (gps 10 hz)
    # identify common timestamp to interpolate to at higher frequency
    commonTime = np.linspace(T_ppk['epochTime'][indsPPK[0]], T_ppk['epochTime'][indsPPK[-1]],
                             int((T_ppk['epochTime'][indsPPK[-1]] - T_ppk['epochTime'][indsPPK[0]]) / .1),
                             endpoint=True)

    f = interpolate.interp1d(S1['time'], sonar_range)  # , fill_value='extrapolate')
    sonar_range_i = f(commonTime)
    f = interpolate.interp1d(T_ppk['epochTime'], T_ppk['height'])
    ppkHeight_i = f(commonTime)
    # now i have both signals at the same time stamps
    phaseLagInSamps, phaseLaginTime = yellowfinLib.findTimeShiftCrossCorr(signal.detrend(ppkHeight_i),
                                                             signal.detrend(sonar_range_i),
                                                             sampleFreq=np.median(np.diff(commonTime)))

    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.plot(S1['time'][sonarIndicies], sonar_range[sonarIndicies], label='sonar_raw')
    plt.plot(T_ppk['epochTime'][indsPPK], T_ppk['GNSS_elevation_NAVD88'][indsPPK], label='ppk elevation NAVD88 m')
    plt.legend()
    plt.subplot(212)
    plt.title(f"sonar data needs to be adjusted by {phaseLaginTime} seconds")
    plt.plot(commonTime, signal.detrend(ppkHeight_i), label='ppk input')
    plt.plot(commonTime, signal.detrend(sonar_range_i), label='sonar input')
    plt.plot(commonTime+phaseLaginTime, signal.detrend(sonar_range_i), '.', label='interp _sonar shifted')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plotDir, 'subsetAfterCrossCorrelation.png'))

    print(f"sonar data adjusted by {phaseLaginTime:.3f} seconds")


    ## now process all data for saving to file
    antenna_offset = 0.25  # meters between the antenna phase center and sounder head
    PPKqualityThreshold = 1
    smoothedSonarConfidence = 60
    sonar_time_out = S1['time'] + phaseLaginTime

    ## ok now put the sonar data on the GNSS timestamps which are decimal seconds.  We can do this with sonar_time_out,
    # because we just adjusted by the phase lag to make sure they are timesynced.
    timeOutInterpStart = np.ceil(sonar_time_out.min() * 10) / 10  # round to nearest 0.1
    timeOutInterpEnd = np.floor(sonar_time_out.max() * 10) / 10  # round to nearest 0.1
    # create a timestamp for data to be output and in phase with that of the ppk gps data which are on the 0.1 s
    time_out = np.linspace(timeOutInterpStart, timeOutInterpEnd, int((timeOutInterpEnd - timeOutInterpStart) / 0.1),
                           endpoint=True)

    print("here's where some better filtering could be done, probably worth saving an intermediate product here for future "
          "revisit")
    print(
        f"for now we're only saving/logging values that have a GNSS fix quality of {PPKqualityThreshold} and a sonar confidence > {smoothedSonarConfidence}")

    # now put relevant GNSS and sonar on output timestamps
    # initalize out variables
    sonar_smooth_depth_out, sonar_smooth_confidence_out = np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan
    sonar_instant_depth_out, sonar_instant_confidence_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan
    sonar_backscatter_out = np.zeros((time_out.shape[0], S1['range_m'].shape[0])) * np.nan
    lat_out, lon_out = np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan
    elevation_out, fix_quality = np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan

    # loop through my common time (.1 s increment) and find associated sonar and gnss values
    # this might be
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
            if T_ppk['Q'][idxTimeMatchGNSS] <= PPKqualityThreshold and S1['smoothed_depth_measurement_confidence'][
                idxTimeMatchSonar] > smoothedSonarConfidence:
                sonar_smooth_depth_out[tidx] = S1['smooth_depth_m'][idxTimeMatchSonar]
                sonar_instant_depth_out[tidx] = S1['this_ping_depth_m'][idxTimeMatchSonar]
                sonar_smooth_confidence_out[tidx] = S1['smoothed_depth_measurement_confidence'][idxTimeMatchSonar]
                sonar_instant_confidence_out[tidx] = S1['this_ping_depth_measurement_confidence'][idxTimeMatchSonar]
                sonar_backscatter_out[tidx] = S1['profile_data'][:, idxTimeMatchSonar]
                lat_out[tidx] = T_ppk['lat'][idxTimeMatchGNSS]
                lon_out[tidx] = T_ppk['lon'][idxTimeMatchGNSS]
                elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                      S1['smooth_depth_m'][
                                          idxTimeMatchSonar]
                fix_quality[tidx] = T_ppk['Q'][idxTimeMatchGNSS]

    # identify data that are not nan's to save
    idxDataToSave = np.argwhere(~np.isnan(sonar_smooth_depth_out))  # identify data that are not NaNs

    fs = 16
    # make a final plot of all the processed data
    plt.figure(figsize=(12,8))
    plt.scatter(lon_out[idxDataToSave], lat_out[idxDataToSave], c=elevation_out[idxDataToSave], label='processed depths')
    cbar = plt.colorbar()
    cbar.set_label('depths NAVD88 [m]', fontsize=fs)
    plt.plot(T_ppk['lon'], T_ppk['lat'], 'k.', ms=0.1, label='vehicle trajectory')
    plt.ylabel('latitude', fontsize=fs)
    plt.xlabel('longitude', fontsize=fs)
    plt.title('final data with elevations', fontsize=fs+4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plotDir, 'FinalDataProduct.png'))


    outputfile = os.path.join(datadir,f'{timeString}_finalDataProduct.h5')
    with h5py.File(outputfile, 'w') as hf:
        hf.create_dataset('time', data=time_out[idxDataToSave])
        hf.create_dataset('longitude', data=lon_out[idxDataToSave])
        hf.create_dataset('latitude', data=lat_out[idxDataToSave])
        hf.create_dataset('elevation', data=elevation_out[idxDataToSave])
        hf.create_dataset('fix_quality', data=fix_quality[idxDataToSave])
        hf.create_dataset('sonar_smooth_depth', data=sonar_smooth_depth_out[idxDataToSave])
        hf.create_dataset('sonar_smooth_confidence', data=sonar_smooth_confidence_out[idxDataToSave])
        hf.create_dataset('sonar_instant_depth', data=sonar_instant_depth_out[idxDataToSave])
        hf.create_dataset('sonar_instant_confidence', data=sonar_instant_confidence_out[idxDataToSave])
        hf.create_dataset('sonar_backscatter_out', data=sonar_backscatter_out[idxDataToSave])



if __name__ == "__main__":
    timeString = "20230417"  # "20230327"

    ## change things in this
    # # establish data location paths to be used for the rest of the code base
    database = r"C:\Users\RDCHLASB\Documents\repos\yellowFin_AcqProcessing-myDev\SampleData"  # "/data/yellowfin/" #"../SampleData"
    # timeString = "20230504"  # "20230327"
    geoidFileLoc = r"C:\Users\RDCHLASB\Documents\repos\yellowFin_AcqProcessing-myDev\g2012bu0.bin"  # https://geodesy.noaa.gov/GEOID/  <-- acquired here, used by pygeodesy
    #
    main(timeString, database, geoidFileLoc)
