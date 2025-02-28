Below, I’ll provide a step-by-step guide with full code snippets to run in Google Colab for extracting features from the CTU-CHB Intrapartum Cardiotocography Database (PhysioNet) to match your dataset’s 21 input variables. I’ll break this into parts, explaining every detail as if you’re starting from scratch, so you can copy-paste each section into a Colab cell and execute it sequentially. Each part builds on the previous one, and I’ll guide you through what’s happening at every step to ensure you understand the process.

---

### Step 0: Set Up Your Google Colab Environment
Before we begin, we need to install the required libraries and prepare the environment. This step ensures you have all the tools to process the CTU-CHB dataset.

#### Code for Step 0
```python
# Step 0: Install necessary libraries
!pip install wfdb  # For reading PhysioNet WFDB files
!pip install numpy  # For numerical operations
!pip install scipy  # For statistical and signal processing functions
!pip install pandas  # For data manipulation and CSV output

# Import the libraries we'll use
import wfdb  # To read CTU-CHB data files
import numpy as np  # For math operations like mean, std, etc.
from scipy import signal  # For filtering and peak detection
from scipy.stats import skew  # For calculating histogram tendency (skewness)
import pandas as pd  # To store and save the extracted features
import matplotlib.pyplot as plt  # For visualizing signals (optional, but helpful)

# Print a message to confirm setup is complete
print("Setup complete! All libraries are installed and imported.")
```

#### Explanation
- **What this does**: This code installs four Python libraries (`wfdb`, `numpy`, `scipy`, `pandas`) that we’ll use to read data, process signals, and save results. Then, it imports them into your Colab environment so they’re ready to use.
- **Why each library**:
  - `wfdb`: Reads the CTU-CHB dataset’s WFDB files (special format for biomedical signals).
  - `numpy`: Handles numbers, arrays, and basic math (e.g., calculating means).
  - `scipy`: Provides advanced signal processing (e.g., filtering noise) and stats (e.g., skewness).
  - `pandas`: Organizes data into a table and saves it as a CSV file.
  - `matplotlib`: Optional, for plotting signals to check our work.
- **What to do**: Copy this code into a new cell in Google Colab, click the “Run” button (play icon), and wait until you see “Setup complete!” in the output. Then move to Step 1.

---

### Step 1: Download and Load a Sample CTU-CHB Recording
Now, we’ll download the CTU-CHB dataset and load one recording to work with. We’ll start with a single file to test our feature extraction before scaling up.

#### Code for Step 1
```python
# Step 1: Download and load a sample CTU-CHB recording

# Define the record number (e.g., 1001 is the first file)
record_name = '1001'

# Download the specific record from PhysioNet
# This command fetches the .dat (signal) and .hea (header) files
wfdb.dl_database('ctu-uhb-ctgdb', dl_dir='ctu_chb_data', records=[record_name])

# Load the record into memory
# 'ctu_chb_data/' is the folder where files are downloaded
record = wfdb.rdrecord(f'ctu_chb_data/{record_name}')

# Extract the signals (FHR and UC) and sampling frequency
fhr_signal = record.p_signal[:, 0]  # Fetal heart rate (first column)
uc_signal = record.p_signal[:, 1]   # Uterine contractions (second column)
fs = record.fs  # Sampling frequency (usually 4 Hz)

# Print some basic info to check the data
print(f"Loaded record: {record_name}")
print(f"Sampling frequency: {fs} Hz")
print(f"FHR signal length: {len(fhr_signal)} samples")
print(f"UC signal length: {len(uc_signal)} samples")

# Optional: Plot the first 1000 samples to visualize
plt.figure(figsize=(12, 6))
plt.plot(fhr_signal[:1000], label='FHR (bpm)')
plt.plot(uc_signal[:1000], label='UC')
plt.title(f'Sample Signal from Record {record_name}')
plt.xlabel('Sample Number')
plt.ylabel('Value')
plt.legend()
plt.show()
```

#### Explanation
- **What this does**: Downloads one CTU-CHB recording (file `1001`) from PhysioNet, loads it, and extracts the FHR and UC signals. It also shows a plot to visualize the raw data.
- **Details**:
  - `wfdb.dl_database`: Downloads files from the CTU-CHB database (`ctu-uhb-ctgdb`) to a folder named `ctu_chb_data`. We specify record `1001`.
  - `wfdb.rdrecord`: Reads the downloaded file into a Python object (`record`).
  - `record.p_signal`: Contains the signals in a 2D array. Column 0 is FHR (in bpm), Column 1 is UC (arbitrary units).
  - `record.fs`: Sampling frequency (should be 4 Hz, meaning 4 samples per second).
  - `plt.plot`: Creates a graph of the first 1000 samples to see what the signals look like.
- **What to expect**: After running, you’ll see text output (e.g., “Sampling frequency: 4 Hz”) and a plot with two lines (FHR and UC). If the plot shows wavy lines, the data loaded correctly.
- **What to do**: Copy this into a new Colab cell, run it, and check the output. If successful, proceed to Step 2.

---

### Step 2: Preprocess the Signals (Noise Filtering)
Raw signals often have noise (random fluctuations). We’ll clean them with a bandpass filter to focus on meaningful CTG patterns.

#### Code for Step 2
```python
# Step 2: Preprocess the signals (noise filtering)

# Define bandpass filter parameters
lowcut = 0.01  # Low frequency cutoff (Hz) to remove slow drifts
highcut = 1.0  # High frequency cutoff (Hz) to remove fast noise
nyquist = fs / 2  # Nyquist frequency (half the sampling rate)

# Design the bandpass filter
b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')

# Apply the filter to FHR and UC signals
fhr_filtered = signal.filtfilt(b, a, fhr_signal)
uc_filtered = signal.filtfilt(b, a, uc_signal)

# Replace NaN or infinite values with interpolation (if any)
fhr_filtered = np.nan_to_num(fhr_filtered, nan=np.nanmean(fhr_filtered))
uc_filtered = np.nan_to_num(uc_filtered, nan=np.nanmean(uc_filtered))

# Print to confirm filtering
print("Signals filtered successfully!")
print(f"First 5 FHR values (filtered): {fhr_filtered[:5]}")
print(f"First 5 UC values (filtered): {uc_filtered[:5]}")

# Optional: Plot filtered vs. raw FHR to compare
plt.figure(figsize=(12, 6))
plt.plot(fhr_signal[:1000], label='Raw FHR', alpha=0.5)
plt.plot(fhr_filtered[:1000], label='Filtered FHR')
plt.title('Raw vs. Filtered FHR')
plt.xlabel('Sample Number')
plt.ylabel('FHR (bpm)')
plt.legend()
plt.show()
```

#### Explanation
- **What this does**: Applies a bandpass filter to remove noise from the FHR and UC signals, keeping only frequencies between 0.01 Hz and 1.0 Hz.
- **Details**:
  - `signal.butter`: Creates a 4th-order Butterworth filter (smooths data effectively) with lowcut (0.01 Hz) and highcut (1.0 Hz), normalized by the Nyquist frequency (2 Hz for 4 Hz sampling).
  - `signal.filtfilt`: Filters the signal twice (forward and backward) for zero-phase distortion, making it cleaner.
  - `np.nan_to_num`: Replaces any NaN (not a number) values with the signal’s mean to avoid errors later.
  - Plot compares raw and filtered FHR to show the smoothing effect.
- **What to expect**: Output shows filtered values (e.g., numbers like `120.5, 121.2`), and the plot shows a smoother FHR line compared to the raw one.
- **What to do**: Copy this into a new cell, run it after Step 1, and check the plot. If the filtered line looks smoother, move to Step 3.

---

### Step 3: Calculate Baseline FHR Iteratively
We need a baseline FHR (average heart rate without accelerations or decelerations) to detect events. This is computed iteratively.

#### Code for Step 3
```python
# Step 3: Calculate Baseline FHR Iteratively

def calculate_baseline_fhr(fhr, fs, max_iter=10, convergence=0.1):
    # Initial baseline is the mean of the entire signal
    baseline = np.mean(fhr)
    window_size = int(10 * 60 * fs)  # 10-minute window in samples (2400 at 4 Hz)

    for _ in range(max_iter):
        # Define accelerations (>15 bpm above baseline for ≥15 sec)
        accel_mask = (fhr >= baseline + 15).astype(int)
        accel_starts = np.where(np.diff(accel_mask) == 1)[0]
        accel_ends = np.where(np.diff(accel_mask) == -1)[0]
        accel_mask_extended = np.zeros_like(fhr)
        for start, end in zip(accel_starts, accel_ends):
            if end - start >= 60:  # 15 sec = 60 samples at 4 Hz
                accel_mask_extended[start:end+1] = 1

        # Define decelerations (<15 bpm below baseline for ≥15 sec)
        decel_mask = (fhr <= baseline - 15).astype(int)
        decel_starts = np.where(np.diff(decel_mask) == 1)[0]
        decel_ends = np.where(np.diff(decel_mask) == -1)[0]
        decel_mask_extended = np.zeros_like(fhr)
        for start, end in zip(decel_starts, decel_ends):
            if end - start >= 60:
                decel_mask_extended[start:end+1] = 1

        # Exclude accelerations and decelerations
        mask = ~(accel_mask_extended.astype(bool) | decel_mask_extended.astype(bool))
        new_baseline = np.mean(fhr[mask])

        # Check for convergence
        if abs(new_baseline - baseline) < convergence:
            break
        baseline = new_baseline

    return baseline

# Calculate the baseline FHR
baseline_fhr = calculate_baseline_fhr(fhr_filtered, fs)

# Print the result
print(f"Baseline FHR: {baseline_fhr:.2f} bpm")
```

#### Explanation
- **What this does**: Computes the baseline FHR by iteratively excluding accelerations and decelerations until the mean stabilizes (changes < 0.1 bpm).
- **Details**:
  - `calculate_baseline_fhr`: A custom function:
    - Starts with the mean of the whole FHR signal.
    - Identifies accelerations (FHR ≥ baseline + 15 bpm for ≥15 sec) and decelerations (FHR ≤ baseline - 15 bpm for ≥15 sec).
    - Creates masks to exclude these segments (using `np.diff` to find start/end points).
    - Recalculates the mean on the remaining data, repeating up to 10 times or until convergence.
  - `window_size`: Represents a 10-minute window (2400 samples at 4 Hz), per ACOG guidelines, though here we apply it iteratively across the whole signal.
- **What to expect**: Output like “Baseline FHR: 135.50 bpm”. This is the steady heart rate without big jumps or drops.
- **What to do**: Copy this into a new cell, run it after Step 2, and note the baseline value. Proceed to Step 4.

---

### Step 4: Detect Events (Accelerations, Decelerations, Contractions)
Now, we’ll count accelerations, classify decelerations, and detect uterine contractions using the baseline.

#### Code for Step 4
```python
# Step 4: Detect Events (Accelerations, Decelerations, Contractions)

# Accelerations
accel_mask = (fhr_filtered >= baseline_fhr + 15).astype(int)
accel_starts = np.where(np.diff(accel_mask) == 1)[0]
accel_ends = np.where(np.diff(accel_mask) == -1)[0]
accelerations = 0
for start, end in zip(accel_starts, accel_ends):
    if end - start >= 60:  # 15 sec = 60 samples
        accelerations += 1
accelerations_per_sec = accelerations / (len(fhr_filtered) / fs)

# Decelerations
decel_mask = (fhr_filtered <= baseline_fhr - 15).astype(int)
decel_starts = np.where(np.diff(decel_mask) == 1)[0]
decel_ends = np.where(np.diff(decel_mask) == -1)[0]
light_decel = severe_decel = prolonged_decel = 0
for start, end in zip(decel_starts, decel_ends):
    duration = (end - start) / fs  # Duration in seconds
    depth = baseline_fhr - np.min(fhr_filtered[start:end+1])
    if duration >= 15:  # Minimum 15 sec
        if duration > 120:  # Prolonged: >2 min
            prolonged_decel += 1
        elif depth > 30:  # Severe: >30 bpm drop
            severe_decel += 1
        elif 15 <= depth <= 30:  # Light: 15-30 bpm drop
            light_decel += 1
light_decel_per_sec = light_decel / (len(fhr_filtered) / fs)
severe_decel_per_sec = severe_decel / (len(fhr_filtered) / fs)
prolonged_decel_per_sec = prolonged_decel / (len(fhr_filtered) / fs)

# Uterine Contractions
uc_mean = np.mean(uc_filtered)
uc_std = np.std(uc_filtered)
uc_threshold = uc_mean + 2 * uc_std  # Threshold for peaks
uc_peaks, _ = signal.find_peaks(uc_filtered, height=uc_threshold, distance=60)
contractions_per_sec = len(uc_peaks) / (len(uc_filtered) / fs)

# Fetal Movement (assumed equal to accelerations)
fetal_movement_per_sec = accelerations_per_sec

# Print results
print(f"Accelerations per sec: {accelerations_per_sec:.6f}")
print(f"Light decelerations per sec: {light_decel_per_sec:.6f}")
print(f"Severe decelerations per sec: {severe_decel_per_sec:.6f}")
print(f"Prolonged decelerations per sec: {prolonged_decel_per_sec:.6f}")
print(f"Uterine contractions per sec: {contractions_per_sec:.6f}")
print(f"Fetal movement per sec: {fetal_movement_per_sec:.6f}")
```

#### Explanation
- **What this does**: Counts events in the signals:
  - **Accelerations**: FHR ≥ baseline + 15 bpm for ≥15 sec.
  - **Decelerations**: FHR ≤ baseline - 15 bpm for ≥15 sec, classified by depth and duration.
  - **Contractions**: Peaks in UC signal above a threshold.
  - **Fetal Movement**: Assumed equal to accelerations (common approximation).
- **Details**:
  - `np.diff`: Finds where FHR crosses thresholds (e.g., +15 bpm).
  - Decelerations split into light (15-30 bpm drop), severe (>30 bpm), prolonged (>2 min).
  - `signal.find_peaks`: Detects UC peaks above mean + 2*std, with a minimum 15-sec gap.
  - Rates computed by dividing counts by total seconds (signal length / fs).
- **What to expect**: Outputs like “Accelerations per sec: 0.000833” (small numbers because they’re per second).
- **What to do**: Copy this into a new cell, run it after Step 3, and note the values. Move to Step 5.

---

### Step 5: Compute Variability Measures
We’ll calculate short-term and long-term variability to match your dataset’s features.

#### Code for Step 5
```python
# Step 5: Compute Variability Measures

# Window size: 1 minute = 240 samples at 4 Hz
window_size = 240

# Short-Term Variability (STV)
stv_values = []
for i in range(0, len(fhr_filtered) - window_size, window_size):
    window = fhr_filtered[i:i + window_size]
    stv = np.mean(np.abs(np.diff(window)))  # Mean of consecutive differences
    stv_values.append(stv)
mean_stv = np.mean(stv_values)
abnormal_stv = 100 * np.mean((np.array(stv_values) < 1) | (np.array(stv_values) > 5))

# Long-Term Variability (LTV)
ltv_values = []
for i in range(0, len(fhr_filtered) - window_size, window_size):
    window = fhr_filtered[i:i + window_size]
    ltv = np.std(window)  # Standard deviation in window
    ltv_values.append(ltv)
mean_ltv = np.mean(ltv_values)
abnormal_ltv = 100 * np.mean((np.array(ltv_values) < 5) | (np.array(ltv_values) > 25))

# Print results
print(f"Mean short-term variability: {mean_stv:.2f} bpm")
print(f"Abnormal short-term variability (%): {abnormal_stv:.2f}")
print(f"Mean long-term variability: {mean_ltv:.2f} bpm")
print(f"Abnormal long-term variability (%): {abnormal_ltv:.2f}")
```

#### Explanation
- **What this does**: Calculates variability over 1-minute windows:
  - **STV**: Average difference between consecutive FHR values (normal: 1-5 bpm).
  - **LTV**: Standard deviation of FHR (normal: 5-25 bpm).
  - **Abnormal %**: Percentage of windows outside normal ranges.
- **Details**:
  - Loops over signal in 1-minute chunks (240 samples).
  - `np.diff`: Computes differences for STV.
  - `np.std`: Computes standard deviation for LTV.
  - Converts abnormal counts to percentages.
- **What to expect**: Outputs like “Mean STV: 2.34 bpm”, “Abnormal STV: 10.50%”.
- **What to do**: Copy this into a new cell, run it after Step 4, and note the values. Proceed to Step 6.

---

### Step 6: Compute Histogram Features
Finally, we’ll create an FHR histogram and extract statistical features.

#### Code for Step 6
```python
# Step 6: Compute Histogram Features

# Create histogram with 1 bpm bins
hist, bin_edges = np.histogram(fhr_filtered, bins=range(int(np.floor(min(fhr_filtered))), int(np.ceil(max(fhr_filtered))) + 1))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Histogram features
histogram_width = max(fhr_filtered) - min(fhr_filtered)
histogram_min = min(fhr_filtered)
histogram_max = max(fhr_filtered)
histogram_peaks, _ = signal.find_peaks(hist, height=1)
histogram_number_of_peaks = len(histogram_peaks)
histogram_number_of_zeroes = np.sum(hist == 0)
histogram_mode = bin_centers[np.argmax(hist)]
histogram_mean = np.mean(fhr_filtered)
histogram_median = np.median(fhr_filtered)
histogram_variance = np.var(fhr_filtered)
histogram_tendency = skew(fhr_filtered)

# Print results
print(f"Histogram width: {histogram_width:.2f}")
print(f"Histogram min: {histogram_min:.2f}")
print(f"Histogram max: {histogram_max:.2f}")
print(f"Number of peaks: {histogram_number_of_peaks}")
print(f"Number of zeroes: {histogram_number_of_zeroes}")
print(f"Mode: {histogram_mode:.2f}")
print(f"Mean: {histogram_mean:.2f}")
print(f"Median: {histogram_median:.2f}")
print(f"Variance: {histogram_variance:.2f}")
print(f"Tendency (skewness): {histogram_tendency:.2f}")
```

#### Explanation
- **What this does**: Builds a histogram of FHR values and computes 10 features.
- **Details**:
  - `np.histogram`: Creates bins every 1 bpm from min to max FHR.
  - `signal.find_peaks`: Counts peaks in the histogram.
  - `skew`: Measures skewness (tendency: negative = left-skewed, 0 = symmetric, positive = right-skewed).
- **What to expect**: Outputs like “Histogram width: 50.00”, “Number of peaks: 3”.
- **What to do**: Copy this into a new cell, run it after Step 5, and note the values. Move to Step 7.

---

### Step 7: Combine Features and Save as CSV
Now, we’ll compile all 21 features into a table and save them.

#### Code for Step 7
```python
# Step 7: Combine Features and Save as CSV

# List of features matching your dataset
features = {
    'baseline_value': baseline_fhr,
    'accelerations': accelerations_per_sec,
    'fetal_movement': fetal_movement_per_sec,
    'uterine_contractions': contractions_per_sec,
    'light_decelerations': light_decel_per_sec,
    'severe_decelerations': severe_decel_per_sec,
    'prolongued_decelerations': prolonged_decel_per_sec,
    'abnormal_short_term_variability': abnormal_stv,
    'mean_value_of_short_term_variability': mean_stv,
    'percentage_of_time_with_abnormal_long_term_variability': abnormal_ltv,
    'mean_value_of_long_term_variability': mean_ltv,
    'histogram_width': histogram_width,
    'histogram_min': histogram_min,
    'histogram_max': histogram_max,
    'histogram_number_of_peaks': histogram_number_of_peaks,
    'histogram_number_of_zeroes': histogram_number_of_zeroes,
    'histogram_mode': histogram_mode,
    'histogram_mean': histogram_mean,
    'histogram_median': histogram_median,
    'histogram_variance': histogram_variance,
    'histogram_tendency': histogram_tendency
}

# Create a DataFrame (one row for this record)
df = pd.DataFrame([features])

# Save to CSV
df.to_csv('ctu_chb_features.csv', index=False)

# Display the DataFrame
print("Extracted features for record 1001:")
print(df)

print("Features saved to 'ctu_chb_features.csv'!")
```

#### Explanation
- **What this does**: Puts all calculated features into a table (DataFrame) with column names matching your original dataset, then saves it as a CSV file.
- **Details**:
  - `features`: A dictionary mapping your feature names to the values we computed.
  - `pd.DataFrame`: Turns the dictionary into a table with one row (for record 1001).
  - `to_csv`: Saves the table to a file you can download from Colab.
- **What to expect**: A table printed with 21 columns and one row, plus a file `ctu_chb_features.csv` in Colab’s file explorer (left sidebar).
- **What to do**: Copy this into a new cell, run it after Step 6, and check the output. You’ve now extracted features for one record!

---

### Step 8: Scale Up to All Records (Optional)
To process all 552 records, we’ll loop through them. This step is optional if you just want to test one record first.

#### Code for Step 8
```python
# Step 8: Process All Records (Optional)

# List of all record names (1001 to 1552)
all_records = [str(i) for i in range(1001, 1553)]

# Download all records (this takes time!)
wfdb.dl_database('ctu-uhb-ctgdb', dl_dir='ctu_chb_data', records=all_records)

# Empty list to store feature rows
all_features = []

for record_name in all_records[:10]:  # Limit to first 10 for testing; remove [:10] for all
    try:
        # Load record
        record = wfdb.rdrecord(f'ctu_chb_data/{record_name}')
        fhr_signal = record.p_signal[:, 0]
        uc_signal = record.p_signal[:, 1]
        fs = record.fs

        # Filter signals
        b, a = signal.butter(4, [0.01 / (fs / 2), 1.0 / (fs / 2)], btype='band')
        fhr_filtered = signal.filtfilt(b, a, fhr_signal)
        uc_filtered = signal.filtfilt(b, a, uc_signal)
        fhr_filtered = np.nan_to_num(fhr_filtered, nan=np.nanmean(fhr_filtered))
        uc_filtered = np.nan_to_num(uc_filtered, nan=np.nanmean(uc_filtered))

        # Calculate all features (reuse previous code)
        baseline_fhr = calculate_baseline_fhr(fhr_filtered, fs)

        accel_mask = (fhr_filtered >= baseline_fhr + 15).astype(int)
        accel_starts = np.where(np.diff(accel_mask) == 1)[0]
        accel_ends = np.where(np.diff(accel_mask) == -1)[0]
        accelerations = sum(1 for s, e in zip(accel_starts, accel_ends) if e - s >= 60)
        accelerations_per_sec = accelerations / (len(fhr_filtered) / fs)

        decel_mask = (fhr_filtered <= baseline_fhr - 15).astype(int)
        decel_starts = np.where(np.diff(decel_mask) == 1)[0]
        decel_ends = np.where(np.diff(decel_mask) == -1)[0]
        light_decel = severe_decel = prolonged_decel = 0
        for start, end in zip(decel_starts, decel_ends):
            duration = (end - start) / fs
            depth = baseline_fhr - np.min(fhr_filtered[start:end+1])
            if duration >= 15:
                if duration > 120:
                    prolonged_decel += 1
                elif depth > 30:
                    severe_decel += 1
                elif 15 <= depth <= 30:
                    light_decel += 1
        light_decel_per_sec = light_decel / (len(fhr_filtered) / fs)
        severe_decel_per_sec = severe_decel / (len(fhr_filtered) / fs)
        prolonged_decel_per_sec = prolonged_decel / (len(fhr_filtered) / fs)

        uc_threshold = np.mean(uc_filtered) + 2 * np.std(uc_filtered)
        uc_peaks, _ = signal.find_peaks(uc_filtered, height=uc_threshold, distance=60)
        contractions_per_sec = len(uc_peaks) / (len(uc_filtered) / fs)
        fetal_movement_per_sec = accelerations_per_sec

        stv_values = [np.mean(np.abs(np.diff(fhr_filtered[i:i + window_size]))) 
                      for i in range(0, len(fhr_filtered) - window_size, window_size)]
        mean_stv = np.mean(stv_values)
        abnormal_stv = 100 * np.mean((np.array(stv_values) < 1) | (np.array(stv_values) > 5))

        ltv_values = [np.std(fhr_filtered[i:i + window_size]) 
                      for i in range(0, len(fhr_filtered) - window_size, window_size)]
        mean_ltv = np.mean(ltv_values)
        abnormal_ltv = 100 * np.mean((np.array(ltv_values) < 5) | (np.array(ltv_values) > 25))

        hist, bin_edges = np.histogram(fhr_filtered, bins=range(int(np.floor(min(fhr_filtered))), int(np.ceil(max(fhr_filtered))) + 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        histogram_width = max(fhr_filtered) - min(fhr_filtered)
        histogram_min = min(fhr_filtered)
        histogram_max = max(fhr_filtered)
        histogram_peaks, _ = signal.find_peaks(hist, height=1)
        histogram_number_of_peaks = len(histogram_peaks)
        histogram_number_of_zeroes = np.sum(hist == 0)
        histogram_mode = bin_centers[np.argmax(hist)]
        histogram_mean = np.mean(fhr_filtered)
        histogram_median = np.median(fhr_filtered)
        histogram_variance = np.var(fhr_filtered)
        histogram_tendency = skew(fhr_filtered)

        # Combine into a feature dictionary
        features = {
            'baseline_value': baseline_fhr,
            'accelerations': accelerations_per_sec,
            'fetal_movement': fetal_movement_per_sec,
            'uterine_contractions': contractions_per_sec,
            'light_decelerations': light_decel_per_sec,
            'severe_decelerations': severe_decel_per_sec,
            'prolongued_decelerations': prolonged_decel_per_sec,
            'abnormal_short_term_variability': abnormal_stv,
            'mean_value_of_short_term_variability': mean_stv,
            'percentage_of_time_with_abnormal_long_term_variability': abnormal_ltv,
            'mean_value_of_long_term_variability': mean_ltv,
            'histogram_width': histogram_width,
            'histogram_min': histogram_min,
            'histogram_max': histogram_max,
            'histogram_number_of_peaks': histogram_number_of_peaks,
            'histogram_number_of_zeroes': histogram_number_of_zeroes,
            'histogram_mode': histogram_mode,
            'histogram_mean': histogram_mean,
            'histogram_median': histogram_median,
            'histogram_variance': histogram_variance,
            'histogram_tendency': histogram_tendency
        }
        all_features.append(features)
        print(f"Processed record: {record_name}")
    except Exception as e:
        print(f"Error processing {record_name}: {e}")

# Create DataFrame and save
df_all = pd.DataFrame(all_features)
df_all.to_csv('ctu_chb_all_features.csv', index=False)
print("All features saved to 'ctu_chb_all_features.csv'!")
```

#### Explanation
- **What this does**: Loops through all 552 records (or 10 for testing), extracts features, and saves them in one CSV.
- **Details**:
  - Combines all previous steps into a loop.
  - `try-except`: Skips errors (e.g., corrupt files) and continues.
  - Limited to 10 records for testing; remove `[:10]` to process all 552 (takes longer).
- **What to expect**: Output like “Processed record: 1001” for each record, then a final CSV file.
- **What to do**: Copy this into a new cell, run it after Step 7, and wait (test with 10 records first). Download `ctu_chb_all_features.csv` from Colab’s file explorer.

---

### Notes
- **Labels**: The CTU-CHB dataset doesn’t have `fetal_health` labels (1, 2, 3). You’ll need clinical outcomes (e.g., pH values) or expert annotation to add them later.
- **Next Steps**: Use this CSV with your TabNet model from Part 2 for external validation. Let me know if you need help with that!
- **Running**: Run each step in order (0 to 7, optionally 8). Check outputs at each step to ensure success.

Copy-paste these into Colab cells, run them one by one, and you’ll have your features ready! Let me know if anything goes wrong or if you need more guidance.
