# ============================================
# THESIS - EEG Motor Imagery Analysis
# Step 1: Load and explore the dataset
# Author: [το όνομά σου]
# Date: April 2026
# ============================================

import mne
import os

# Διαδρομή του dataset
DATA_PATH = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "BNP-My Thesis", "me matzako", "dataset")

# Διαδρομή του πρώτου αρχείου
file_path = os.path.join(DATA_PATH, "S001", "S001R03.edf")

# Φόρτωση του αρχείου
print("Loading EEG file...")
raw = mne.io.read_raw_edf(file_path, preload=True)

# Εκτύπωση βασικών πληροφοριών
print("\n--- EEG File Information ---")
print(raw.info)
print(f"\nNumber of channels: {len(raw.ch_names)}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds")
print("\nFile loaded successfully!")