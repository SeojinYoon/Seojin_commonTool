
# MARK: - Common Libraries
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# MARK: - Functions
def sampling(start_sampling_timing, sampling_interval, sampling_count, timings, values):
    """
    Sampling data
    
    Each value has corresponding timing. 
    
    timings:
        |-----------|
        0(s)        1(s)
        
    values:
        |-----------|
        10          100
        
    If i want to sample more data from values, we need to interpolate value from given values.
    In case of bounding timing, sampling value is assigned by adjacent value.
    
    ex)
    If i set sampling interval: 0.2, sampling count: 6, start_sampling_timing: 0
                |--|--|--|--|--|--|
    timing:     0                 1
    values:     10                100
    timing(s):  0 0.2 0.4 0.6 0.8 1
    values(s):  10, 28.0, 46.0, 64.0, 82.0, 100
    
    :param start_sampling_timing: (float) - start sampling timing
    :param sampling_interval: sampling interval(float) - sampling interval
    :param sampling_count: (int) - how many sample do you need
    :param timings: timings(list) - timing data
    :param values: values(list) - corresponding value from each timing
    """
    n_interval_decimal = len(str(float(sampling_interval)).split(".")[1])
    end_sampling_timing = np.round(start_sampling_timing + sampling_count * sampling_interval, n_interval_decimal)
    sampling_timings = np.arange(start_sampling_timing, 
                                 end_sampling_timing, 
                                 sampling_interval)

    sampling_values = []
    for sample_timing in sampling_timings:
        if sample_timing in timings:
            index = timings.index(sample_timing)
            sampling_values.append(values[index])
        else:
            previous_idx = find_nearest(array = timings, value = sample_timing, check_type = "previous")
            next_idx = find_nearest(array = timings, value = sample_timing, check_type = "next")
            if previous_idx != None and next_idx != None:
                total_timing_length = np.abs(timings[next_idx] - timings[previous_idx])
                diff_timing_fromPrevious = np.abs(sample_timing - timings[previous_idx])
                diff_timing_fromNext = np.abs(timings[next_idx] - sample_timing)

                previous_weight = diff_timing_fromNext / total_timing_length
                next_weight = diff_timing_fromPrevious / total_timing_length

                # interpolation
                previous_value = values[previous_idx]
                next_value = values[next_idx]

                interpolate_value = previous_value * previous_weight + next_value * next_weight

                sampling_values.append(interpolate_value)
            elif previous_idx == None and next_idx != None:
                sampling_values.append(values[next_idx])
            elif previous_idx != None and next_idx == None:
                sampling_values.append(values[previous_idx])

    return sampling_values
    
def downsampling(df: pd.DataFrame,
                 time_interval: float,
                 kind: str = "linear") -> pd.DataFrame:
    """
    Do downsampling over each column using time information with interpolation.

    :param df: The dataframe must include a 'times' column.
    :param kind: Specifies the kind of interpolation as a string ('linear', 'cubic', etc.).

    :return result: The downsampled DataFrame
    """
    x = df["times"].values
    y = df.drop(columns=["times"]).values

    new_times = np.arange(x.min(), x.max() + 1e-7, time_interval)
    new_times = np.clip(new_times, x.min(), x.max())

    # Interpolation
    f = interp1d(x, y, kind=kind, axis=0, bounds_error=False, fill_value=(y[0], y[-1]))
    new_y = f(new_times)

    # Make dataframe
    new_df = pd.DataFrame(new_y, columns=df.columns.drop("times"))
    new_df.insert(0, "times", new_times)
    
    return new_df

# MARK: - Examples
if __name__ == "__main__":
    sampling(start_sampling_value = 0, 
         sampling_interval = 0.01, 
         sampling_count = 10, 
         timings = [0.0020409011840820113, 0.013019599914550761, 0.024019994735717753, 0.03501991271972654, 0.04601983070373533, 0.05602053642272947, 0.06702045440673826, 0.07793525695800779, 0.08801988601684568, 0.09802011489868162], 
         values = [0,1,2,3,4,5,6,7,8,9])
    
    sampling(start_sampling_value = 0, 
         sampling_interval = 0.2, 
         sampling_count = 6, 
         timings = [0,1], 
         values = [10, 100])