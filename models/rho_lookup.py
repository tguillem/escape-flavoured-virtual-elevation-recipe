import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Union
from datetime import datetime

class RhoLookup:
    def __init__(self, rho_path: str,
                 window_analysis_samples: int = 500,
                 target_noise_reduction: float = 0.8,
                 max_window: int = 300):
        """
        Initialize air density calculator with adaptive window calculation.

        Parameters:
        -----------
        rho_path : str
            Path to the CSV file
        window_analysis_samples : int
            Number of samples to use for window optimization (default: 500)
        target_noise_reduction : float
            Target noise reduction factor (0-1, default: 0.8 = 80% noise reduction)
        max_window : int
            Maximum allowed window size in samples (default: 300)
        """
        self.target_noise_reduction = target_noise_reduction
        self.max_window = max_window
        self.analysis_info = {}  # Store all analysis information

        # Read and preprocess data
        df = self._read_data(rho_path)

        # Calculate optimal windows based on data characteristics
        windows = self._calculate_optimal_windows(
            df[['Temperature', 'Station Pressure', 'Relative Humidity']].values[:window_analysis_samples],
            df.index[:window_analysis_samples]
        )

        self.temp_window = windows['Temperature']
        self.pa_window = windows['Pressure']
        self.rh_window = windows['Humidity']

        # Apply moving averages
        df['Temperature_smooth'] = df['Temperature'].rolling(
            window=self.temp_window, center=True, min_periods=1
        ).mean()
        df['Relative Humidity_smooth'] = df['Relative Humidity'].rolling(
            window=self.rh_window, center=True, min_periods=1
        ).mean()
        df['Station Pressure_smooth'] = df['Station Pressure'].rolling(
            window=self.pa_window, center=True, min_periods=1
        ).mean()

        # Calculate air density for both raw and smoothed data
        df = self._calculate_density(df)
        df = self._calculate_density(df, "_smooth")

        # Resample to 1-second intervals
        df_resampled = df[['rho', 'rho_smooth']].resample('1s').interpolate()
        df_resampled = df_resampled.reset_index()[['timestamp', 'rho', 'rho_smooth']]

        self.rho_df = df_resampled

    def _read_data(self, rho_path: str) -> pd.DataFrame:
        """Read and parse the CSV file."""
        # Find header row
        with open(rho_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith("FORMATTED DATE_TIME"):
                    header_row_index = i
                    break
            else:
                raise ValueError("No valid header line found in file.")

        def skiprows_func(x):
            return x < header_row_index or x == header_row_index + 1

        # Read CSV
        df = pd.read_csv(rho_path, skiprows=skiprows_func)
        df['timestamp'] = pd.to_datetime(df['FORMATTED DATE_TIME'])
        df = df.set_index('timestamp')
        df = df[['Temperature', 'Station Pressure', 'Relative Humidity']]

        return df

    def _calculate_optimal_windows(self, data: np.ndarray, timestamps: pd.DatetimeIndex) -> Dict[str, int]:
        """
        Calculate optimal moving average windows based on signal characteristics.

        Returns dict with optimal window sizes for each parameter.
        """
        # Calculate sampling interval
        intervals = np.diff(timestamps).astype('timedelta64[s]').astype(float)
        avg_interval = np.mean(intervals)
        self.analysis_info['sampling_interval'] = avg_interval

        windows = {}
        param_names = ['Temperature', 'Pressure', 'Humidity']
        self.analysis_info['parameters'] = {}

        for i, param_name in enumerate(param_names):
            values = data[:, i]

            noise_level = self._estimate_noise(values)

            autocorr = self._calculate_autocorrelation(values, max_lag=200)

            decorr_time = self._find_decorrelation_time(autocorr, threshold=0.5)

            signal_std = np.std(values)
            snr = signal_std / noise_level if noise_level > 0 else np.inf

            optimal_window = self._optimize_window(
                values, 
                decorr_time, 
                snr,
                self.target_noise_reduction
            )

            windows[param_name] = optimal_window

            self.analysis_info['parameters'][param_name] = {
                'noise_level': noise_level,
                'signal_std': signal_std,
                'snr': snr,
                'decorrelation_time': decorr_time,
                'optimal_window': optimal_window,
                'optimal_window_seconds': optimal_window * avg_interval
            }

        return windows

    def _estimate_noise(self, values: np.ndarray) -> float:
        """Estimate measurement noise from consecutive differences."""
        diffs = np.abs(np.diff(values))
        # Use median absolute deviation for robust noise estimation
        mad = np.median(diffs)
        # Convert MAD to standard deviation equivalent
        noise_std = mad * 1.4826
        return noise_std

    def _calculate_autocorrelation(self, values: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        # Remove mean
        values_centered = values - np.mean(values)
        # Use numpy's correlation function
        autocorr = np.correlate(values_centered, values_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        return autocorr[:max_lag]

    def _find_decorrelation_time(self, autocorr: np.ndarray, threshold: float = 0.5) -> int:
        """Find the lag where autocorrelation drops below threshold."""
        below_threshold = np.where(autocorr < threshold)[0]
        if len(below_threshold) > 0:
            return below_threshold[0]
        return len(autocorr)

    def _optimize_window(self, values: np.ndarray, decorr_time: int, 
                        snr: float, target_reduction: float) -> int:
        """
        Determine optimal window size based on signal characteristics.
        """
        # Base window on decorrelation time
        base_window = max(3, decorr_time // 4)

        # Adjust based on SNR
        if snr < 5:  # Very noisy
            window_multiplier = 2.0
        elif snr < 10:  # Moderately noisy
            window_multiplier = 1.5
        elif snr < 20:  # Some noise
            window_multiplier = 1.2
        else:  # Clean signal
            window_multiplier = 1.0

        # Test windows to achieve target noise reduction
        test_windows = range(max(3, int(base_window * 0.5)), 
                           min(self.max_window, int(base_window * window_multiplier * 3)))

        best_window = base_window
        original_noise = self._estimate_noise(values)

        for window in test_windows[::5]:  # Test every 5th window for efficiency
            smoothed = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean()
            remaining_noise = self._estimate_noise(smoothed.values)

            if original_noise > 0:
                reduction = 1 - (remaining_noise / original_noise)
                if reduction >= target_reduction:
                    best_window = window
                    break

        # Ensure window is odd for centered averaging
        return best_window if best_window % 2 == 1 else best_window + 1

    def _calculate_density1(self, df: pd.DataFrame, suffix="") -> pd.DataFrame:
        """
        Calculate moist air density using exact formula.

        Formula: ρ = (p_d/R_d + p_v/R_v) / T
        where p_d = p_total - p_v (partial pressure of dry air)
        """
        # Constants
        R_d = 287.05    # J/(kg·K) - specific gas constant for dry air
        R_v = 461.5     # J/(kg·K) - specific gas constant for water vapor

        # Temperature conversion
        df['Temperature_K' + suffix] = df['Temperature' + suffix] + 273.15
        df['Pressure_Pa' + suffix] = df['Station Pressure' + suffix] * 100

        # Calculate saturation vapor pressure (Magnus formula - your version)
        T_C = df['Temperature' + suffix]
        df['p_sat' + suffix] = 0.61078 * np.exp(17.27 * T_C / (T_C + 237.3))  # kPa

        # Calculate actual vapor pressure
        phi = df['Relative Humidity' + suffix] / 100.0  # Convert % to fraction
        df['p_v_kPa' + suffix] = phi * df['p_sat' + suffix]  # kPa
        df['p_v_Pa' + suffix] = df['p_v_kPa' + suffix] * 1000  # Convert to Pa

        # Calculate partial pressure of dry air
        df['p_d_Pa' + suffix] = df['Pressure_Pa' + suffix] - df['p_v_Pa' + suffix]

        # Calculate moist air density using exact formula
        T_K = df['Temperature_K' + suffix]
        rho_d = df['p_d_Pa' + suffix] / (R_d * T_K)  # Density contribution from dry air
        rho_v = df['p_v_Pa' + suffix] / (R_v * T_K)  # Density contribution from water vapor

        df['rho' + suffix] = rho_d + rho_v

        return df

    def _calculate_density(self, df: pd.DataFrame, suffix="") -> pd.DataFrame:
        """
        Calculate moist air density

        Formula: ρ = (p_d/R_d + p_v/R_v) / T
        where p_d = p - p_v (partial pressure of dry air)
        p_v = φ * p_sat (partial pressure of water vapor)
        and p_sat = 0.61078 * exp(17.27 * T_C / (T_C + 237.3)) (saturation vapor pressure)
        """
        # Constants
        R_d = 287.05    # J/(kg·K) - specific gas constant for dry air
        R_v = 461.5     # J/(kg·K) - specific gas constant for water vapor

        # Extract input data
        T_C = df['Temperature' + suffix].values  # °C
        T_K = T_C + 273.15  # K
        P_Pa = df['Station Pressure' + suffix].values * 100  # Pa
        RH = df['Relative Humidity' + suffix].values  # %

        # Calculate saturation vapor pressure
        p_sat_kPa = 0.61078 * np.exp(17.27 * T_C / (T_C + 237.3))  # kPa

        # Calculate partial pressure of water vapor
        phi = RH / 100.0
        p_v_Pa = phi * p_sat_kPa * 1000

        # Calculate partial pressure of dry air
        p_d_Pa = P_Pa - p_v_Pa

        # Calculate moist air density
        # ρ = p_d/(R_d*T) + p_v/(R_v*T)
        rho_d = p_d_Pa / (R_d * T_K)  # Density contribution from dry air
        rho_v = p_v_Pa / (R_v * T_K)  # Density contribution from water vapor

        # Store only final result in DataFrame
        df['rho' + suffix] = rho_d + rho_v

        return df

    def get_df(self) -> pd.DataFrame:
        """Return the processed dataframe with timestamps and air density."""
        return self.rho_df

    def get_window_params(self) -> Dict[str, int]:
        """Return the calculated window parameters."""
        return {
            'temp_window': self.temp_window,
            'pa_window': self.pa_window,
            'rh_window': self.rh_window
        }

    def get_analysis_info(self) -> Dict[str, any]:
        """Return all analysis information collected during initialization."""
        return self.analysis_info

    def get_rho_at_timestamp(self, timestamp: Union[str, datetime, pd.Timestamp]) -> Optional[Tuple[float, float]]:
        """
        Get both raw and smoothed air density at a specific timestamp.

        Parameters:
        -----------
        timestamp : str, datetime, or pd.Timestamp
            The timestamp to query. Can be:
            - String in ISO format (e.g., '2024-01-15 14:30:00')
            - datetime object
            - pandas Timestamp
            Note: Since data is resampled to 1-second intervals, timestamp should be at second precision

        Returns:
        --------
        tuple of (float, float) or None
            (raw_rho, smoothed_rho) in kg/m³ at the given timestamp, or None if timestamp is out of range
        """
        # Convert input to pandas Timestamp
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp)

        # Round to nearest second since data is at 1-second intervals
        timestamp = timestamp.round('1s')

        # Check if timestamp is within data range
        if timestamp < self.rho_df['timestamp'].min() or timestamp > self.rho_df['timestamp'].max():
            return None

        # Find the exact match (should always exist since we have 1-second intervals)
        match = self.rho_df[self.rho_df['timestamp'] == timestamp]
        if not match.empty:
            return (float(match['rho'].iloc[0]), float(match['rho_smooth'].iloc[0]))

        # This should not happen if timestamp is properly rounded to seconds
        return None

    def get_rho_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the time range of available air density data.

        Returns:
        --------
        tuple of (pd.Timestamp, pd.Timestamp)
            Start and end timestamps of the data
        """
        return (self.rho_df['timestamp'].min(), self.rho_df['timestamp'].max())

    def get_rho_mean_between(self, 
                           start_time: Union[str, datetime, pd.Timestamp], 
                           end_time: Union[str, datetime, pd.Timestamp, None] = None,
                           duration_seconds: Optional[float] = None,
                           use_minmax_only: bool = False) -> Optional[Dict[str, float]]:
        """
        Calculate mean air density between two timestamps.

        Parameters:
        -----------
        start_time : str, datetime, or pd.Timestamp
            Start timestamp for the calculation
        end_time : str, datetime, pd.Timestamp, or None
            End timestamp. If None, duration_seconds must be provided
        duration_seconds : float or None
            Duration in seconds from start_time. Used if end_time is None
        use_minmax_only : bool
            If True, calculate mean using only min and max values in the range
            If False (default), calculate mean using all points

        Returns:
        --------
        dict or None
            Dictionary containing:
            - 'mean_raw': Mean of raw air density
            - 'mean_smooth': Mean of smoothed air density
            - 'start_time': Actual start time used
            - 'end_time': Actual end time used
            - 'n_points': Number of points used in calculation
            - 'min_rho': Minimum air density in range (for minmax calculation)
            - 'max_rho': Maximum air density in range (for minmax calculation)
            Returns None if time range is invalid
        """
        # Convert start_time to pandas Timestamp
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        elif isinstance(start_time, datetime):
            start_time = pd.Timestamp(start_time)

        # Determine end_time
        if end_time is None and duration_seconds is None:
            raise ValueError("Either end_time or duration_seconds must be provided")

        if end_time is None:
            end_time = start_time + pd.Timedelta(seconds=duration_seconds)
        else:
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
            elif isinstance(end_time, datetime):
                end_time = pd.Timestamp(end_time)

        # Round to nearest second
        start_time = start_time.round('1s')
        end_time = end_time.round('1s')

        # Ensure start_time is before end_time
        if start_time > end_time:
            start_time, end_time = end_time, start_time

        # Check if time range is within data bounds
        data_start, data_end = self.get_rho_time_range()
        if start_time > data_end or end_time < data_start:
            return None

        # Clip to available data range
        start_time = max(start_time, data_start)
        end_time = min(end_time, data_end)

        # Get data slice
        mask = (self.rho_df['timestamp'] >= start_time) & (self.rho_df['timestamp'] <= end_time)
        data_slice = self.rho_df[mask]

        if data_slice.empty:
            return None

        result = {
            'start_time': start_time,
            'end_time': end_time,
            'n_points': len(data_slice)
        }

        if use_minmax_only:
            # Calculate mean using only min and max values
            min_rho = data_slice['rho'].min()
            max_rho = data_slice['rho'].max()
            min_rho_smooth = data_slice['rho_smooth'].min()
            max_rho_smooth = data_slice['rho_smooth'].max()

            result.update({
                'mean_raw': (min_rho + max_rho) / 2,
                'mean_smooth': (min_rho_smooth + max_rho_smooth) / 2,
                'min_rho': min_rho,
                'max_rho': max_rho,
                'min_rho_smooth': min_rho_smooth,
                'max_rho_smooth': max_rho_smooth,
                'method': 'minmax'
            })
        else:
            # Calculate mean using all points
            result.update({
                'mean_raw': data_slice['rho'].mean(),
                'mean_smooth': data_slice['rho_smooth'].mean(),
                'min_rho': data_slice['rho'].min(),
                'max_rho': data_slice['rho'].max(),
                'min_rho_smooth': data_slice['rho_smooth'].min(),
                'max_rho_smooth': data_slice['rho_smooth'].max(),
                'method': 'all_points'
            })

        return result


# Example usage
if __name__ == "__main__":
    import sys
    from datetime import timedelta

    if len(sys.argv) < 2:
        print("Usage: python rho_lookup.py <csv_file> [timestamp]")
        print("Example: python rho_lookup.py 19406901625_ACTIVITY.csv")
        print("Example with timestamp: python rho_lookup.py 19406901625_ACTIVITY.csv '2024-01-15 14:30:00'")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        print(f"Processing file: {csv_file}")
        rho_lookup = RhoLookup(csv_file)

        # Check for range or duration queries
        if len(sys.argv) > 3:
            start_time = sys.argv[2]
            duration = float(sys.argv[3])

            print(f"\nCalculating mean air density from {start_time} for {duration} seconds")

            for use_minmax in [True, False]:
                result = rho_lookup.get_rho_mean_between(start_time, duration_seconds=duration, use_minmax_only=use_minmax)
                if result:
                    print(f"\nResults: {'min/max only' if use_minmax else 'all points'}")
                    print(f"  Time range: {result['start_time']} to {result['end_time']}")
                    print(f"  Number of points: {result['n_points']}")
                    print(f"  Mean air density (raw): {result['mean_raw']:.4f} kg/m³")
                    print(f"  Mean air density (smooth): {result['mean_smooth']:.4f} kg/m³")
                    print(f"  Range (raw): {result['min_rho']:.4f} to {result['max_rho']:.4f} kg/m³")
                    print(f"  Range (smooth): {result['min_rho_smooth']:.4f} to {result['max_rho_smooth']:.4f} kg/m³")
                else:
                    print("Error: Invalid time range")
            sys.exit(0)

        # Single timestamp query
        if len(sys.argv) > 2:
            query_timestamp = sys.argv[2]
            print(f"\nQuerying air density at: {query_timestamp}")
            result = rho_lookup.get_rho_at_timestamp(query_timestamp)
            if result is not None:
                rho_raw, rho_smooth = result
                print(f"Air density at {query_timestamp}:")
                print(f"  Raw (instant):    {rho_raw:.4f} kg/m³")
                print(f"  Smoothed (MA):    {rho_smooth:.4f} kg/m³")
                print(f"  Difference:       {abs(rho_smooth - rho_raw):.4f} kg/m³ ({abs(rho_smooth - rho_raw)/rho_raw*100:.2f}%)")
            else:
                print(f"Timestamp {query_timestamp} is outside the data range!")
            sys.exit(0)

        # Get all analysis information
        analysis_info = rho_lookup.get_analysis_info()

        # Print analysis details
        print("\nAnalyzing data characteristics...")
        print(f"Average sampling interval: {analysis_info['sampling_interval']:.1f} seconds")

        for param_name, info in analysis_info['parameters'].items():
            print(f"\n{param_name}:")
            print(f"  Noise level: {info['noise_level']:.4f}")
            print(f"  Signal std: {info['signal_std']:.4f}")
            print(f"  SNR: {info['snr']:.2f}")
            print(f"  Decorrelation time: {info['decorrelation_time']} samples")
            print(f"  Optimal window: {info['optimal_window']} samples ({info['optimal_window_seconds']:.1f} seconds)")

        # Get the processed data
        df = rho_lookup.get_df()
        print(f"\nProcessed {len(df)} data points")

        # Get time range
        start_time, end_time = rho_lookup.get_rho_time_range()
        print(f"Time range: {start_time} to {end_time}")

        # Get the window parameters that were calculated
        windows = rho_lookup.get_window_params()
        print(f"\nOptimal windows determined:")
        print(f"  Temperature: {windows['temp_window']} samples")
        print(f"  Pressure: {windows['pa_window']} samples")
        print(f"  Humidity: {windows['rh_window']} samples")

        # Show some statistics
        print(f"\nAir density statistics:")

        print(f"\nSmoothed (Moving Average):")
        print(f"  Mean: {df['rho_smooth'].mean():.3f} kg/m³")
        print(f"  Std:  {df['rho_smooth'].std():.3f} kg/m³")
        print(f"  Min:  {df['rho_smooth'].min():.3f} kg/m³")
        print(f"  Max:  {df['rho_smooth'].max():.3f} kg/m³")

        # Demo: Query some example timestamps
        print("\nExample queries:")
        # Query at start, middle, and end
        mid_time = start_time + (end_time - start_time) / 2

        for desc, ts in [("Start", start_time),
                        ("Middle", mid_time),
                        ("End", end_time),
                        ("Start + 30 min", start_time + timedelta(minutes=30))]:
            result = rho_lookup.get_rho_at_timestamp(ts)
            if result is not None:
                rho_raw, rho_smooth = result
                print(f"  {desc} ({ts}):")
                print(f"    Raw:      {rho_raw:.4f} kg/m³")
                print(f"    Smoothed: {rho_smooth:.4f} kg/m³")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
