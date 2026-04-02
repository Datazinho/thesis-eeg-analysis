# ============================================
# THESIS - EEG Motor Imagery Analysis
# Step 2: Quality check of all 109 subjects
# Author: [το όνομά σου]
# Date: April 2026
# ============================================

import mne
import os

# Διαδρομή του dataset
DATA_PATH = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "BNP-My Thesis", "me matzako", "dataset")

# Runs που μας ενδιαφέρουν
RUNS = ["R03", "R04", "R05", "R06", "R07", "R08",
        "R09", "R10", "R11", "R12", "R13", "R14"]

# Λίστες για αποθήκευση αποτελεσμάτων
valid_subjects = []
invalid_subjects = []

print("Starting quality check for all 109 subjects...")
print("=" * 50)

# Έλεγχος κάθε subject
for i in range(1, 110):
    subject = f"S{i:03d}"
    subject_path = os.path.join(DATA_PATH, subject)
    has_error = False

    for run in RUNS:
        file_path = os.path.join(subject_path, f"{subject}{run}.edf")

        # Έλεγχος αν υπάρχει το αρχείο
        if not os.path.exists(file_path):
            has_error = True
            break

        # Προσπάθεια φόρτωσης
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            if len(raw.ch_names) != 64:
                has_error = True
                break
        except:
            has_error = True
            break

    if has_error:
        invalid_subjects.append(subject)
        print(f"❌ {subject} - INVALID")
    else:
        valid_subjects.append(subject)
        print(f"✅ {subject} - valid")

# Αποτελέσματα
print("\n" + "=" * 50)
print(f"Valid subjects: {len(valid_subjects)}")
print(f"Invalid subjects: {len(invalid_subjects)}")
print(f"Invalid list: {invalid_subjects}")