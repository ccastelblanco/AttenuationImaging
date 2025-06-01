import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import geopandas as gpd
from scipy.stats import tmean, tstd
from geopy import distance
from geopy.distance import distance
import numpy as np

from utils import *



# Methods to include: plot_grid, compute_3d_attenuation, plot_attenuation, plot_ellipsoid, 
# plot_profile

class AttenuationImager:
    """
    Class for calculating attenuation distribution in a given region in 3D space using
    ellipsoids and plotting slices of the resulting matrices.
    """
    def __init__(self, region: list[float], cell_interval: int,  max_depth: int,  df: pd.DataFrame, dict_df_cols : dict, freqs: list[float], years: list[int], v: float, step: float|None = None): 
        """
        Initialize the class with the region, cell interval, step size, max depth and DataFrame.

        Parameters
        ----------
        region : list 
            List of geographic coordinates defining the region [xmin, xmax, ymin, ymax].
        cell_interval: float 
            Interval in km for creating cells in the grid.            
        max_depth : float
            Maximum depth in km for the grid.
        df : DataFrame
            DataFrame containing attenuation data. The DataFrame must have the following columns: event time, event latitude, event longitude, event depth (m), coda window time (s), station latitude, station longitude, station altitude (m), frequency, error, Qi^-1, Qsc^-1, Qt^-1.
        dict_df_cols : dict
            Dictionary containing column names in the DataFrame. The keys must be: 'event time', 'event_latitude', 'event_longitude', 'event_depth', 'coda_window_time', 'station_latitude', 'station_longitude', 'station_altitude', 'frequency', 'error', 'Qi^-1', 'Qsc^-1', 'Qt^-1'.
        freqs : list
            List of frequencies to be used in the calculations.
        years : list
            List of years to be used in the calculations.
        v : float
            S wave velocity in km/s.
        step : float, optional
            Step size in km for the grid. If None, the step size is set to 1/4 of the cell interval.
        

        """
        self.region = region
        self.cell_interval = cell_interval
        self.__cell_interval_deg = cell_interval/111.1  # Convert km to degrees for x_array and y_array
        if step:
            self.__step_deg = step/111.1 # Convert km to degrees for input step
            self.step = self.__step_deg * 111.1  # Convert degrees to km for step
        else:
            self.step = self.__cell_interval_deg/4*111.1 # By default, step is 1/4 of the cell interval in km
        self.x_array = np.arange(region[0], region[1], self.__cell_interval_deg)
        self.y_array = np.arange(region[2], region[3], self.__cell_interval_deg)
        self.z_array = np.arange(0, max_depth+self.step, self.step)/111.1  # Convert km to degrees for z_array
        self.xv, self.yv, self.zv = np.meshgrid(self.x_array, self.y_array, self.z_array) # Create a grid of coordinates x,y,z from the width, length and height of the area
        self.dx = (self.x_array[1]-self.x_array[0])/2
        self.dy = (self.y_array[1]-self.y_array[0])/2
        self.extent = [self.x_array[0]-self.dx, self.x_array[-1]+self.dx, self.y_array[0]-self.dy, self.y_array[-1]+self.dy]
        self.df = df
        self.dict_df_cols = dict_df_cols
        
        # Assign attributes to column names from df_cols
        self.ev_time_col = self.dict_df_cols['event_time']
        self.ev_lat_col = self.dict_df_cols['event_latitude']
        self.ev_lon_col = self.dict_df_cols['event_longitude']
        self.ev_depth_col = self.dict_df_cols['event_depth']
        self.ctw_col = self.dict_df_cols['coda_window_time']
        self.st_lat_col = self.dict_df_cols['station_latitude']
        self.st_lon_col = self.dict_df_cols['station_longitude']
        self.st_alt_col = self.dict_df_cols['station_altitude']
        self.freq_col = self.dict_df_cols['frequency']
        self.error_col = self.dict_df_cols['error']
        self.Qi_col = self.dict_df_cols['Qi^-1']
        self.Qsc_col = self.dict_df_cols['Qsc^-1']
        self.Qt_col = self.dict_df_cols['Qt^-1']
        
        # Attenuation attributes
        self.freqs = freqs
        self.years = years
        self.v = v
        
        
    def plot_grid(self):
        """
        Plot the grid of coordinates in 3D space.
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plotted grid.
        """       
             
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        
        ax.scatter(self.xv, self.yv, self.zv)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_zlabel('Depth (km)')     # type: ignore
        ax.invert_zaxis()  # type: ignore # Invert z-axis to show depth correctly
        ax.set_box_aspect(aspect = (1,1,1)) # type: ignore
        
        
        ax.zaxis.set_major_formatter(plt.FuncFormatter(z_fmt)) # type: ignore
        ax.zaxis.set_major_locator(plt.MultipleLocator(20/111.1)) # type: ignore
        
        cubic_box(self.xv, self.yv, self.zv, ax)
        # fig.show()
    
    def compute_3d_attenuation(self, lims: dict, save_path: str|None = None):           
        """
        Compute the 3D attenuation distribution using ellipsoids and plot slices of the resulting matrices.

        Parameters
        ----------
        lims : dict
            Dictionary containing for each frequency the limits for calculating mean and std of Qi^-1 and Qsc^-1 based on the log10(Qi^-1) distribution.
        save_path : str, optional
            Path to save the resulting DataFrame. If None, the DataFrame is not saved.
        
        Returns
        -------
        Dataframe 
            DataFrame containing the computed 3D attenuation distribution.
        
        """
        
        
        self.df.dropna(inplace=True)  # Drop NaN values from the DataFrame  
             
        q_types = ['Qi^-1', 'Qsc^-1', 'Qt^-1']        
        
        # Calculating columns for log 10(Qi^-1), log10(Qsc^-1) and log10(Qt^-1) ass values follow a log distribution
        
        for q_type in q_types:
            self.df[f'log10({q_type})'] = np.log10(self.df[q_type])
              
        # Parametrization ellipsoid precomputing parameters
        n = 256
        gamma = np.linspace(0, 2 * np.pi, n)
        beta = np.linspace(0, np.pi, n)
        gamma, beta = np.meshgrid(gamma, beta) # Create a grid of angles for the ellipsoid parametrization
        
        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        ones_gamma = np.ones_like(gamma)
        
        xyz_ellipsoids_coordinates = {}
        
        attenuation_data = {}
        
        for year in self.years:
            print(f'Starting to compute {year}')
            df_year = self.df[self.df[self.ev_time_col].str.contains(str(year), regex=False)]
            
            xyz_ellipsoids_coordinates[year] = {}
            attenuation_data[year] = {}
            
            for f in self.freqs:
                
                xyz_ellipsoids_coordinates[year][f] = {}
                attenuation_data[year][f] = {}
                
                df_freq = df_year[df_year[self.freq_col] == f] # Filter dataframe by frequency     
                              
                lim = lims[f]    # Limits for calculating mean and std of Qi^-1 and Qsc^-1 based on the log10(Qi^-1) distribution    
                
                log10_Qi_nonan_tmean = tmean(df_freq['log10(Qi^-1)'], limits=lim)
                log10_Qsc_nonan_tmean = tmean(df_freq['log10(Qsc^-1)'], limits=lim)
                log10_Qt_nonan_tmean = tmean(df_freq['log10(Qt^-1)'], limits=lim)
                log10_Qi_nonan_tstd = tstd(df_freq['log10(Qi^-1)'], limits=lim)
                log10_Qsc_nonan_tstd = tstd(df_freq['log10(Qsc^-1)'], limits=lim)
                log10_Qt_nonan_tstd = tstd(df_freq['log10(Qt^-1)'], limits=lim)

                # Z score for Qi^-1, Qsc^-1 and Qt^-1 to filter outliers by standard deviation: (x - mean)/std 
                z_score_Qi = (df_freq['log10(Qi^-1)'] - log10_Qi_nonan_tmean) / log10_Qi_nonan_tstd
                z_score_Qsc = (df_freq['log10(Qsc^-1)'] - log10_Qsc_nonan_tmean) / log10_Qsc_nonan_tstd
                z_score_Qt = (df_freq['log10(Qt^-1)'] - log10_Qt_nonan_tmean) / log10_Qt_nonan_tstd

                # Filter dataframes by z score
                df_freq_filter_Qi = df_freq[abs(z_score_Qi) <= 3]
                df_freq_filter_Qsc = df_freq[abs(z_score_Qsc) <= 3]
                df_freq_filter_Qt = df_freq[abs(z_score_Qt) <= 3]
                
                # wave_n = v/f # Wavelength in km
                
                dframes_freq_Q = {'Qi': df_freq_filter_Qi, 'Qsc': df_freq_filter_Qsc, 'Qt': df_freq_filter_Qt}
                
                for key, dframe in dframes_freq_Q.items():
                    
                    xyz_ellipsoids_coordinates[year][f][key] = []
                    ms_freq_Q = []
                    
                    for row in dframe.iterrows():

                        _, data = row
                        
                        # Event coordinates                
                        x1 = data[self.ev_lon_col]
                        y1 = data[self.ev_lat_col]
                        z1 = (data[self.ev_depth_col]/1000)/111.1
                        
                        # Station coordinates                
                        x2 = data[self.st_lon_col]
                        y2 = data[self.st_lat_col]
                        z2 = -(data[self.st_alt_col]/1000)/111.1
                        
                        x0 = (x1+x2)/2
                        y0 = (y1+y2)/2
                        z0 = (z1+z2)/2
                        
                        t = data[self.ctw_col]  # Coda window time in seconds
                        
                        a, b, theta, phi = compute_ellipsoid_parameters(x1, y1, x2, y2, z1, z2, self.v, t)  # Get the ellipsoid parameters with semi-axis lengths a, b, b
                        
                        if b == 0:  # If b is 0, skip the iteration
                            continue                        

                        x_rot, y_rot, z_rot = translate_and_rotate_points(theta, phi, x0, y0, z0, self.xv, self.yv, self.zv)  # Get the rotated and translated coordinates of the grid points
                        
                        # Evaluate the ellipsoid equation to determine if points are inside or outside the ellipsoid
                        zs = (x_rot**2 / (a / 111.1)**2) + (y_rot**2 / (b / 111.1)**2) + (z_rot**2 / (b / 111.1)**2)                  

                        zs[zs<=1] = 1  # Assign value of 1 to all points inside the ellipsoid
                        zs[zs>1] = np.nan   # Assign NaN to all points outside the ellipsoid
                        zs1 = zs.copy()                        
                        
                        # ---------------------------------------------------------------------------------------
                        
                        ####### Parametrization of the ellipsoid to get its coordinates #######
                        x, y, z = parameterize_ellipsoid(a, b, b, sin_beta, cos_beta, cos_gamma, sin_gamma, ones_gamma)  # Get the ellipsoid coordinates with semi-axis lengths a, b, b
                        
                        
                        # Obtain rotated and translated ellipsoid points
                        xrt, yrt, zrt = matrix_rot_trans(x, y, z, 0, phi, -theta, x0, y0, z0)
                        xyz_ellipsoids_coordinates[year][f][key].append((xrt, yrt,zrt))
                        
                        # ---------------------------------------------------------------------------------------
                                            
                        # Assign attenuation values to the points inside the ellipsoid
                        if key == 'Qi':
                            zs1[zs1==1] = data['log10(Qi^-1)']
                        elif key == 'Qsc':
                            zs1[zs1==1] = data['log10(Qsc^-1)']
                        else:
                            zs1[zs1==1] = data['log10(Qt^-1)']
                        
                        ms_freq_Q.append(zs1)
          
                    ms_freq_Q = np.array(ms_freq_Q)
                    ms_freq_Q_mean = np.nanmean(ms_freq_Q, axis=0)
                    ms_freq_Q_std = np.nanstd(ms_freq_Q, axis=0, ddof=1)
                    

                    if key == 'Qi':
                        attenuation_data[year][f]['Qi^-1_mean'] = ms_freq_Q_mean
                        attenuation_data[year][f]['Qi^-1_std'] = ms_freq_Q_std
                    elif key == 'Qsc':
                        attenuation_data[year][f]['Qsc^-1_mean'] = ms_freq_Q_mean
                        attenuation_data[year][f]['Qsc^-1_std'] = ms_freq_Q_std
                    else:
                        attenuation_data[year][f]['Qt^-1_mean'] = ms_freq_Q_mean
                        attenuation_data[year][f]['Qt^-1_std'] = ms_freq_Q_std                        


                print(f'{f} Hz finished')                          
            
            print(f'{year} finished\n')
        
        
        self.xyz_ellipsoids_coordinates = xyz_ellipsoids_coordinates
             
        # Saving the arrays into a file

        xv_flat = self.xv.flatten()
        yv_flat = self.yv.flatten()
        zv_km = self.zv*111.1
        zv_km_flat = zv_km.flatten()

        df_years = []

        for year in attenuation_data.keys():
            
            df_freqs = []
            
            for freq in attenuation_data[year].keys():
        
                arr_df = pd.DataFrame()

                arr_df['x'] = xv_flat
                arr_df['y'] = yv_flat
                arr_df['z'] = zv_km_flat
                
                arr_df['Qi^-1'] = 10**attenuation_data[year][freq]['Qi^-1_mean'].flatten()
                arr_df['Qsc^-1'] = 10**attenuation_data[year][freq]['Qsc^-1_mean'].flatten()
                arr_df['Qt^-1'] = 10**attenuation_data[year][freq]['Qt^-1_mean'].flatten()

                arr_df['Qi^-1_std'] = 10**attenuation_data[year][freq]['Qi^-1_std'].flatten()
                arr_df['Qsc^-1_std'] = 10**attenuation_data[year][freq]['Qt^-1_mean'].flatten()
                arr_df['Qt^-1_std'] = 10**attenuation_data[year][freq]['Qt^-1_mean'].flatten()
                
                arr_df['frequency'] = freq
                
                df_freqs.append(arr_df)

            df_year = pd.concat(df_freqs, ignore_index = True)
            df_year['year'] = year
            df_years.append(df_year)

        df_all = pd.concat(df_years, ignore_index = True)
        self.df_3d_calcs = df_all.copy()
        
        if save_path:
            df_all.to_csv(save_path, index=False)
            
        return df_all
  
    
    def plot_attenuation(self, df_3d_calcs: pd.DataFrame, slice_depth: float, im: np.ndarray|None = None, gdfs: list[gpd.GeoDataFrame]|None = None, save_path: str|None = None):
        """
        Plot maps of the attenuation distribution for a given depth slice.
        
        Parameters
        ----------
        df_3d_calcs : DataFrame
            DataFrame containing the 3D attenuation data.
        slice_depth : float
            Depth in km for the slice to plot.
        im : array
            Image to plot as background.
        gdfs (list, optional): List of GeoDataFrames to plot on top of the image. GeoDataFrames must be in the same coordinate system as the image and ordered as the desired display. 
        save_path (str, optional): Path to save the resulting plots. If None, the plots are not saved.
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plotted maps.

        """
                 

        # Iterating over years and frequencies to plot the maps
        
        for year in self.years:
            print(year)

            fig, spec = fig_setup(self.freqs)

            for i, f in enumerate(self.freqs):

                axes = fig_axes_config(fig, spec, i, self.extent, self.region, im=im, gdfs=gdfs)
                
                filtered_df = df_3d_calcs[
                    (df_3d_calcs['year'] == year) &
                    (df_3d_calcs['frequency'] == f) &
                    (df_3d_calcs['z'] == slice_depth)
                ]

                Q_arrays = [
                    reshape_and_flip(filtered_df['Qi^-1'].values),
                    reshape_and_flip(filtered_df['Qsc^-1'].values),
                    reshape_and_flip(filtered_df['Qt^-1'].values)
                ]

                for ax, data in zip(axes, Q_arrays):
                    im_ = ax.imshow(data, extent=self.extent, vmin=data.min(), vmax=data.max(),
                                    alpha=0.7, interpolation='bicubic', zorder=4)
                    cb = fig.colorbar(im_, ax=ax, shrink=1)
                    cb.formatter.set_powerlimits((0,0)) # type: ignore
                    cb.formatter.set_useMathText(True) # type: ignore
                    cb.ax.yaxis.set_offset_position('left')

                axes[0].set_ylabel(f'{f} Hz', rotation=0, labelpad=30)

            fig.suptitle(f'{year}\nDepth: 0 km', fontsize=14, y=0.96)

            if save_path:
                fig.savefig(f'{save_path}/{year}_Q^-1_freqs_{slice_depth}km_slice.png', dpi=300, bbox_inches='tight')

            print('done\n')
            
    def compute_attenuation_differences(self, df_3d_calcs: pd.DataFrame, save_path: str|None = None):
        """
        Compute the differences in attenuation for each frequency and year.

        Parameters
        ----------
        df_3d_calcs : DataFrame
            DataFrame containing the 3D attenuation data.
        save_path : str, optional
            Path to save the resulting DataFrame. If None, the DataFrame is not saved.
        
        Returns
        -------
        DataFrame
            DataFrame containing the differences in attenuation for each frequency and year.
        list
            List containing the max and min Qi^-1 difference values for each frequency across years.
        list
            List containing the max and min Qsc^-1 difference values for each frequency across years.
        list
            List containing the max and min Qt^-1 difference values for each frequency across years.

        """
        
        # Compute max and min per year and frequency
        def compute_extremes(diffs, axis=(0, 1)):
            max_years, min_years = [], []
            for year_diffs in diffs:
                max_freqs = [d[:,:,0].max() for d in year_diffs]
                min_freqs = [d[:,:,0].min() for d in year_diffs]
                max_years.append(max_freqs)
                min_years.append(min_freqs)
            return max_years, min_years
        
        # Compute max/min per frequency across years
        def max_min_across_years(max_years, min_years):
            max_freqs = np.max(max_years, axis=0)
            min_freqs = np.min(min_years, axis=0)
            return max_freqs, min_freqs
           
        # DataFrame grouping by year and frequency
        df_grouped = df_3d_calcs.groupby(['year', 'frequency'])
        
        # Differences storage
        ms_diffs = {'Qi': [], 'Qsc': [], 'Qt': []}
        
        for year in self.years:
            if year == 2022:
                break
            
            year_diffs = {'Qi': [], 'Qsc': [], 'Qt': []}
            
            for f in self.freqs:
                
                # Get the DataFrame for the current year and frequency
                df1 = df_grouped.get_group((year, f)) 
                df2 = df_grouped.get_group((year+1, f))

                # Calculate the differences in Qi^-1, Qsc^-1 and Qt^-1 and reshape them to the grid shape
                Qi_diff = (df2['Qi^-1'].to_numpy() - df1['Qi^-1'].to_numpy()).reshape(self.xv.shape)
                Qsc_diff = (df2['Qsc^-1'].to_numpy() - df1['Qsc^-1'].to_numpy()).reshape(self.xv.shape)
                Qt_diff = (df2['Qt^-1'].to_numpy() - df1['Qt^-1'].to_numpy()).reshape(self.xv.shape)
                
                # Append the differences to the year_diffs dictionary
                year_diffs['Qi'].append(Qi_diff)
                year_diffs['Qsc'].append(Qsc_diff)
                year_diffs['Qt'].append(Qt_diff)
            
            # Append the year_diffs to the ms_diffs dictionary based on attenuation type keys
            for key in ms_diffs:
                ms_diffs[key].append(year_diffs[key])
                
        Qi_max_years, Qi_min_years = compute_extremes(ms_diffs['Qi'])
        Qsc_max_years, Qsc_min_years = compute_extremes(ms_diffs['Qsc'])
        Qt_max_years, Qt_min_years = compute_extremes(ms_diffs['Qt'])
        
        Qi_max_freqs_years, Qi_min_freqs_years = max_min_across_years(Qi_max_years, Qi_min_years)
        Qsc_max_freqs_years, Qsc_min_freqs_years = max_min_across_years(Qsc_max_years, Qsc_min_years)
        Qt_max_freqs_years, Qt_min_freqs_years = max_min_across_years(Qt_max_years, Qt_min_years)
        
        # Create DataFrames for differences
        xv_flat, yv_flat, zv_km_flat = self.xv.flatten(), self.yv.flatten(), (self.zv * 111.1).flatten()

        df_diffs = []

        for diffs, year in zip(zip(ms_diffs['Qi'], ms_diffs['Qsc'], ms_diffs['Qt']), self.years):
            Qi_diffs, Qsc_diffs, Qt_diffs = diffs
            df_freqs = []

            for i, f in enumerate(self.freqs):
                df_f = pd.DataFrame({
                    'Qi^-1_diff': Qi_diffs[i].flatten(),
                    'Qsc^-1_diff': Qsc_diffs[i].flatten(),
                    'Qt^-1_diff': Qt_diffs[i].flatten(),
                    'x': xv_flat,
                    'y': yv_flat,
                    'z': zv_km_flat,
                    'frequency': f
                })
                df_freqs.append(df_f)

            df_diff = pd.concat(df_freqs, ignore_index=True)
            df_diff['difference'] = f"{year}-{year+1}"
            df_diffs.append(df_diff)

        df_all = pd.concat(df_diffs, ignore_index=True)
        
        if save_path:
            df_all.to_csv(save_path, index=False)
              
        return df_all, [Qi_max_freqs_years, Qi_min_freqs_years], [Qsc_max_freqs_years, Qsc_min_freqs_years], [Qt_max_freqs_years, Qt_min_freqs_years]
    
    def plot_attenuation_differences(self, df_diffs: pd.DataFrame, ):
        """
        Plot the differences in attenuation for each frequency and pairs of years.

        Parameters
        ----------
        df_diffs : DataFrame
            DataFrame containing the differences in attenuation data.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plotted maps.
            
        """
        
        for i, year in enumerate(self.years[:-1]):
            print(f'Diferencia {self.years[i]}-{self.years[i + 1]}')

            fig, spec = fig_setup(self.freqs)
        
           
    def plot_ellipsoids(self, year: int, freq: float, ctw: int|None = None):
        """
        Plot ellipsoidal regions in the 3D grid from the event-station pairs.
        
        Parameters
        ----------
        year : int, optional
            Year to filter the DataFrame. If None, all years are used.
        freq : float, optional
            Frequency to filter the DataFrame. If None, all frequencies are used.
        ctw : int, optional
            Coda window time to filter the DataFrame. If None, all coda window times are used.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing maps of the plotted ellipsoids.
        
        """
        if year:
            df_year = self.df[self.df[self.ev_time_col].str.contains(str(year), regex=False)]
            df = df_year[(df_year[self.freq_col] == freq)&(df_year[self.ctw_col] == ctw)]
        else:
            df = self.df[(self.df[self.freq_col] == freq)&(self.df[self.ctw_col] == ctw)]
        
        fig = plt.figure(figsize=(10,10))  # Square figure
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect(aspect = (1,1,1)) # type: ignore
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_zlabel('Depth (km)') # type: ignore
        
        # ellipsoids_coordinates = self.xyz_ellipsoids_coordinates[year][freq][q_type]  # Get the ellipsoids coordinates for the given year, frequency and attenuation type

        # for ellipsoid in ellipsoids_coordinates:
            
        #     x = ellipsoid[0]
        #     y = ellipsoid[1]
        #     z = ellipsoid[2]     
                   
        count = 0

        for index,  row in df.iterrows():
            #  print(row['id_evento'])
            if np.isnan(row[self.Qi_col]):
                continue
            # Event coordinates
            x1 = row[self.ev_lon_col]
            y1 = row[self.ev_lat_col]
            z1 = (row[self.ev_depth_col]/1000)/111.1
            
            # Station coordinates  
            x2 = row[self.st_lon_col]
            y2 = row[self.st_lat_col]
            z2 = -(row[self.st_alt_col]/1000)/111.1
            
            # print(x1,x2,y1,y2)
            
            # Ellipse parameters
            # Ellipse equation: x**2/(v*t/2)**2 + y**2/((v*t/2)**2 - r**2/4))  
            
            x0 = (x1+x2)/2
            y0 = (y1+y2)/2
            z0 = (z1+z2)/2
            t = row[self.ctw_col] 
            r = (((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5)*111.1 # Event-station distance in km
            a = self.v*t/2 # Major semiaxis in km
            b = (a**2 - r**2/4)**0.5 # Minor semiaxis in km
            
            if np.iscomplex(b):
                continue
            
            slope_xy = (y2-y1)/(x2-x1) # Slope of the line between event-station in the xy-plane
            theta = -np.arctan(slope_xy)
            x1_p = x1/np.cos(theta)
            x2_p = x2/np.cos(theta)
            slope_xz = (z2-z1)/(x2_p-x1_p) # Slope of the line between event-station in the xz-plane
            phi = -np.arctan(slope_xz)
            # slope_yz = (z2-z1)/(y2-y1) #Pendiente de recta entre evento-estación en plano yz
            # alpha = -np.arctan(slope_yz)
            # print('h:',h,'k:',k,'t:',t,'f:',f,'r:',r,'v:',v,'a:',a,'b:',b,'slope:',slope,'theta:',theta)

            gamma = np.linspace(0,2*np.pi, 256).reshape(256, 1) # the angle of the projection in the xy-plane
            beta = np.linspace(0, np.pi, 256).reshape(-1, 256) # the angle from the polar axis, ie the polar angle
            gamma, beta = np.meshgrid(gamma, beta) # Create a grid of angles for the ellipsoid parametrization
                        
            # Ellipsoid parametrization
            x = (a/111)*np.sin(beta)*np.cos(gamma)
            y = (b/111)*np.sin(beta)*np.sin(gamma)
            z = (b/111)*np.cos(beta)*np.ones_like(gamma) 
            
            xrt, yrt, zrt = matrix_rot_trans(x, y, z, 0, phi, -theta, x0, y0, z0)
            
            ax.plot_surface(xrt, yrt, zrt, color='g', alpha=0.2) # type: ignore
            x_f = [x1,x2]
            y_f = [y1,y2] 
            z_f = [z1,z2]
            ax.scatter(x_f,y_f,z_f, c='k')
            
            count +=1
            
            # print(x_f,y_f)
            # print(row)
        
        ax.zaxis.set_major_formatter(plt.FuncFormatter(z_fmt)) # type: ignore
        ax.zaxis.set_major_locator(plt.MultipleLocator(20/111.1)) # type: ignore

        # plt.axis('off')

    # def plot_ellipses_maps(self, slice_depth: float, im: np.ndarray = None, gdfs: list[gpd.GeoDataFrame] = None, save_path: str = None):
    #     """
    #     Plot maps of the projected ellipses for a given depth slice.
        
    #     Args:
    #         df_3d_calcs (DataFrame): DataFrame containing the 3D attenuation data.
    #         slice_depth (float): Depth in km for the slice to plot.
    #         im (array): Image to plot as background.
    #         gdfs (list, optional): List of GeoDataFrames to plot on top of the image. GeoDataFrames must be in the same coordinate system as the image and ordered as the desired display. 
    #         save_path (str, optional): Path to save the resulting plots. If None, the plots are not saved.
    #     """

    #     df = self.df
        
    #     # Calculate log10(Q^-1) column

    #     q_types = ['Qi^-1', 'Qsc^-1', 'Qt^-1']
        
    #     for q_type in q_types:
    #         df[f'log10({q_type})'] = np.log10(df[q_type])

    #     for year in self.years:
    #         print(year)
            
    #         fig, spec = fig_setup(self.freqs)        

    #         for i, f in enumerate(self.freqs):
                
    #             axes = fig_axes_config(fig, spec, i, self.extent, self.region, im=im, gdfs=gdfs)

    #             # df_freq = df_year[df_year['frecuencia'] == f] # Filter dataframe by frequency
                
    #             filtered_df = df[
    #                 (df['year'] == year) &
    #                 (df['frequency'] == f)
    #             ]
                
    #             log10_Q_tmean = {}
    #             log10_Q_tstd = {}
                
    #             for q in q_types:
    #                 log10_Q_tmean[q] = tmean(filtered_df[f'log10({q})'], limits=lims[f])
    #                 log10_Q_tstd[q] = tstd(filtered_df[f'log10({q})'], limits=lims[f])
                    
                    
                    
                
    #             # Z score for Qi^-1, Qsc^-1 and Qt^-1 to filter outliers by standard deviation: (x - mean)/std        
    #             # lim = lims[f]        
                
    #             # log10_Qi_nonan_tmean = tmean(df_freq['log10(Qi^-1)'], limits=lim)
    #             # log10_Qsc_nonan_tmean = tmean(df_freq['log10(Qsc^-1)'], limits=lim)
    #             # log10_Qt_nonan_tmean = tmean(df_freq['log10(Qt^-1)'], limits=lim)
    #             # log10_Qi_nonan_tstd = tstd(df_freq['log10(Qi^-1)'], limits=lim)
    #             # log10_Qsc_nonan_tstd = tstd(df_freq['log10(Qsc^-1)'], limits=lim)
    #             # log10_Qt_nonan_tstd = tstd(df_freq['log10(Qt^-1)'], limits=lim)

    #             z_score_Qi = (df_freq['log10(Qi^-1)'] - log10_Qi_nonan_tmean) / log10_Qi_nonan_tstd
    #             z_score_Qsc = (df_freq['log10(Qsc^-1)'] - log10_Qsc_nonan_tmean) / log10_Qsc_nonan_tstd
    #             z_score_Qt = (df_freq['log10(Qt^-1)'] - log10_Qt_nonan_tmean) / log10_Qt_nonan_tstd

    #             # Filter dataframes by z score
    #             df_freq_filter_Qi = df_freq[abs(z_score_Qi) <= 3]
    #             df_freq_filter_Qsc = df_freq[abs(z_score_Qsc) <= 3]
    #             df_freq_filter_Qt = df_freq[abs(z_score_Qt) <= 3]
                        

    #             axes = [ax1, ax2, ax3]

    #             dframes_freq_Q = [df_freq_filter_Qi, df_freq_filter_Qsc, df_freq_filter_Qt]
                                    
    #             for i, dframe in enumerate(dframes_freq_Q):
                    
    #                 count = 0
    #                 # fig = plt.figure(figsize=(10,10))  # Square figure
    #                 # ax = fig.add_subplot(111, projection='3d')
                                
    #                 for row in dframe.iterrows():

    #                     index, data = row
                        
    #                     # Event coordinates                
    #                     x1 = data['long_evento']
    #                     y1 = data['lat_evento']
    #                     z1 = (data['prof_evento (m)']/1000)/111.1
                        
    #                     # Station coordinates                
    #                     x2 = data['long_estacion']
    #                     y2 = data['lat_estacion']
    #                     z2 = -(data['alt_estacion']/1000)/111.1
                        
    #                     x0 = (x1+x2)/2
    #                     y0 = (y1+y2)/2
    #                     z0 = (z1+z2)/2
                        
    #                     t = data['ventana_tiempo_coda (s)']
    #                     r = distance((y1,x1),(y2,x2)).km
                        
    #                     a = (v*t/2)
    #                     b = (a**2 - r**2/4)**0.5
                        
    #                     if np.iscomplex(b):
    #                         continue
                        
    #                     slope_xy = (y2-y1)/(x2-x1)
    #                     theta = -np.arctan(slope_xy)    # Rotation angle in xy plane
    #                     x1_p = x1/np.cos(theta)
    #                     x2_p = x2/np.cos(theta)
    #                     slope_xz = (z2-z1)/(x2_p-x1_p)
    #                     phi = -np.arctan(slope_xz)
                        
    #                     # x_f = [x1,x2]  
    #                     # y_f = [y1,y2]
                        
    #                     # Parametrization of the ellipsoid
    #                     gamma = np.linspace(0, 2*np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
    #                     beta = np.linspace(0, np.pi, 256).reshape(-1, 256)    # the angle from the polar axis, i.e., the polar angle
    #                     x = (a/111) * np.sin(beta) * np.cos(gamma) # x-coordinates of the ellipsoid
    #                     y = (b/111) * np.sin(beta) * np.sin(gamma) # y-coordinates of the ellipsoid
    #                     z = (b/111) * np.cos(beta) * np.ones_like(gamma) # z-coordinates of the ellipsoid. It is multiplied 
                        
    #                     # Obtain rotated and translated ellipsoid points
    #                     xrt, yrt, zrt = matrix_rot_trans(x, y, z, 0, phi, -theta, x0, y0, z0)
                        
    #                     z_depth_contour = 0 # Desired z depth for contour
    #                     axes[i].contour(xrt, yrt, zrt, [z_depth_contour], colors='r', alpha=0.5, linewidths=0.5, zorder=4)
    #                     # axes[i].scatter(x_f,y_f)

    #                     count += 1
    #                     # if count == 100:
    #                     #     break
    #                 # break
    #             if index_Qi == 0:
    #                 ax1.set_title("$\mathdefault{Qi^{-1}}$")
    #             if index_Qsc == 1:    
    #                 ax2.set_title("$\mathdefault{Qsc^{-1}}$")
    #             if index_Qt == 2:
    #                 ax3.set_title("$\mathdefault{Qt^{-1}}$")        

    #             ax1.set_ylabel(str(f) + ' Hz', rotation=0, labelpad=30)
                
    #             index_Qi += 3
    #             index_Qsc += 3
    #             index_Qt += 3
                
                
            #     break
            # break
            # fig.suptitle(f'{year}\nProfundidad: 0 km', fontsize=14, y=0.96)            
            # fig.savefig(fr"E:\Maestría Geofísica\Tesis\Atenuación sísmica\figuras\{year}_sampling_ellipses_{z_depth_contour}km", dpi=300, bbox_inches='tight')
            # plt.close(fig)
        
        
if __name__ == '__main__':
    print('Modules loaded')
    # pass
    
    # Example usage:
    # df = pd.read_csv(r"E:\Maestría Geofísica\Tesis\Atenuación sísmica\calculos_atenuacion_evento-estacion_2016-2022_elevcorr.csv")

    # dict_df_cols = {
    #     'event_time': 'tiempo',
    #     'event_latitude': 'lat_evento',
    #     'event_longitude': 'long_evento',
    #     'event_depth': 'prof_evento (m)',
    #     'coda_window_time': 'ventana_tiempo_coda (s)',
    #     'station_latitude': 'lat_estacion',
    #     'station_longitude': 'long_estacion',
    #     'station_altitude': 'alt_estacion',
    #     'frequency': 'frecuencia',
    #     'error': 'error',
    #     'Qi^-1': 'Qi^-1',
    #     'Qsc^-1': 'Qsc^-1',
    #     'Qt^-1': 'Qt^-1'
    # }

    # region = [-74.4, -72.9, 6.7, 7.8]  # Example region
    # cell_interval = 10  # Example cell interval in km
    # max_depth = 20  # Example maximum depth in km

    # freqs = [0.525, 1.5, 3.0, 6.0, 12.0, 24.0] # Example frequencies in Hz
    # years = [2016, 2017, 2018, 2019, 2020, 2021, 2022] # Example years
    # v = 3.5  # Example S wave velocity in km/s

    # # Creating an AttenuationImager object
    # at = AttenuationImager(region, cell_interval, max_depth, df=df, dict_df_cols=dict_df_cols, freqs=freqs, years=years, v=v)
    
    # # Running the 3D calculations
    
    # lims = {0.525:[-3.40,None], # Limits to trim outliers to calculate mean and std
    #     1.5:[-3.85,None],
    #     3:[-4.15,None],
    #     6:[-4.45,None],
    #     12:[-4.75,None],
    #     24:[-5.05,None]}

    # df_3d_calcs = at.compute_3d_attenuation(lims)
    
    # at.plot_ellipsoids(year=2016, freq=0.525, q_type='Qi')  # Example of plotting ellipsoids for a specific year, frequency and attenuation type
    
    

    
    
    