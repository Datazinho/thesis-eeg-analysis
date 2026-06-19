# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step_quality_check_v2.py
# Σκοπός  : Αυστηρότερος ποιοτικός έλεγχος για 109 subjects
#           Ελέγχει MINIMUM και MAXIMUM annotations
# Έκδοση  : v2 (αυστηρότερη από v1)
# ============================================================
# ΔΙΑΦΟΡΑ ΑΠΟ V1:
# Η v1 έλεγχε μόνο αν τα annotations είναι ΛΙΓΟΤΕΡΑ
# από το minimum (T0<10, T1<5, T2<5)
#
# Η v2 ελέγχει ΕΠΙΣΗΣ αν τα annotations είναι ΠΕΡΙΣΣΟΤΕΡΑ
# από το maximum (T0>17, T1>12, T2>12)
#
# Αυτό αποκλείει subjects όπως S088, S092 που έχουν
# T0=19, T1=9-10 — ασυνήθιστα υψηλός αριθμός trials
# (σύμφωνα με Shuqfa et al., 2024)
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import mne
import os
import pandas as pd

# ============================================================
# ΒΗΜΑ 1: Παράμετροι
# ============================================================

DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"

RUNS_TO_CHECK = ["R03", "R04", "R05", "R06", "R07", "R08",
                 "R09", "R10", "R11", "R12", "R13", "R14"]

# ── Όρια annotations ──────────────────────────────────────
# MINIMUM: κάτω από αυτό = ελλιπή δεδομένα
MIN_T0 = 10
MIN_T1 = 5
MIN_T2 = 5

# MAXIMUM: πάνω από αυτό = ασυνήθιστα πολλά trials
# Βάσει Shuqfa et al. (2024) και της δομής του dataset
MAX_T0 = 17
MAX_T1 = 12
MAX_T2 = 12

# Όρια διάρκειας (δευτερόλεπτα)
MIN_DURATION = 60    # τουλάχιστον 60 δευτερόλεπτα
MAX_DURATION = 200   # το πολύ 200 δευτερόλεπτα

print("=" * 65)
print("QUALITY CHECK v2 — 109 Subjects (Αυστηρός Έλεγχος)")
print("=" * 65)
print(f"\nΌρια annotations:")
print(f"  T0: {MIN_T0} ≤ T0 ≤ {MAX_T0}")
print(f"  T1: {MIN_T1} ≤ T1 ≤ {MAX_T1}")
print(f"  T2: {MIN_T2} ≤ T2 ≤ {MAX_T2}")
print(f"\nΞεκινάει ο έλεγχος...")
print("-" * 65)

# ============================================================
# ΒΗΜΑ 2: Λίστες αποτελεσμάτων
# ============================================================
valid_subjects   = []
invalid_subjects = []
all_results      = []

# ============================================================
# ΒΗΜΑ 3: Έλεγχος κάθε subject
# ============================================================
for i in range(1, 110):
    subject      = f"S{i:03d}"
    subject_path = os.path.join(DATA_PATH, subject)

    result = {
        'subject'      : subject,
        'folder_exists': False,
        'all_runs_ok'  : True,
        'annotations_ok': True,
        'problems'     : [],
        'valid'        : False
    }

    # Έλεγχος φακέλου
    if not os.path.exists(subject_path):
        result['folder_exists'] = False
        result['all_runs_ok']   = False
        result['problems'].append("Δεν βρέθηκε ο φάκελος")
        invalid_subjects.append(subject)
        all_results.append(result)
        print(f"❌ {subject}: Δεν βρέθηκε ο φάκελος!")
        continue

    result['folder_exists'] = True

    # Έλεγχος runs
    missing_runs = []
    for run in RUNS_TO_CHECK:
        file_path = os.path.join(subject_path, f"{subject}{run}.edf")
        if not os.path.exists(file_path):
            missing_runs.append(run)

    if missing_runs:
        result['all_runs_ok'] = False
        result['problems'].append(f"Λείπουν runs: {missing_runs}")

    # Έλεγχος annotations για R03 και R05
    annotation_problems = []

    for run in ["R03", "R05"]:
        file_path = os.path.join(subject_path, f"{subject}{run}.edf")

        if not os.path.exists(file_path):
            continue

        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

            # Κανάλια
            n_channels = len(raw.ch_names)
            if n_channels != 64:
                annotation_problems.append(
                    f"{run}: {n_channels} κανάλια (αναμένεται 64)")

            # Διάρκεια
            duration = raw.times[-1]
            if duration < MIN_DURATION:
                annotation_problems.append(
                    f"{run}: διάρκεια {duration:.0f}s (αναμένεται ≥{MIN_DURATION}s)")
            if duration > MAX_DURATION:
                annotation_problems.append(
                    f"{run}: διάρκεια {duration:.0f}s (αναμένεται ≤{MAX_DURATION}s)")

            # Annotations
            annotations = raw.annotations
            count_T0 = sum(1 for d in annotations.description if 'T0' in d)
            count_T1 = sum(1 for d in annotations.description if 'T1' in d)
            count_T2 = sum(1 for d in annotations.description if 'T2' in d)

            result[f'{run}_T0'] = count_T0
            result[f'{run}_T1'] = count_T1
            result[f'{run}_T2'] = count_T2
            result[f'{run}_channels'] = n_channels
            result[f'{run}_duration'] = round(duration, 1)

            # ── Έλεγχος MINIMUM ──
            if count_T0 < MIN_T0:
                annotation_problems.append(
                    f"{run}: T0={count_T0} < {MIN_T0} (πολύ λίγα)")
            if count_T1 < MIN_T1:
                annotation_problems.append(
                    f"{run}: T1={count_T1} < {MIN_T1} (πολύ λίγα)")
            if count_T2 < MIN_T2:
                annotation_problems.append(
                    f"{run}: T2={count_T2} < {MIN_T2} (πολύ λίγα)")

            # ── Έλεγχος MAXIMUM (νέο στη v2!) ──
            if count_T0 > MAX_T0:
                annotation_problems.append(
                    f"{run}: T0={count_T0} > {MAX_T0} (πολύ πολλά!)")
            if count_T1 > MAX_T1:
                annotation_problems.append(
                    f"{run}: T1={count_T1} > {MAX_T1} (πολύ πολλά!)")
            if count_T2 > MAX_T2:
                annotation_problems.append(
                    f"{run}: T2={count_T2} > {MAX_T2} (πολύ πολλά!)")

        except Exception as e:
            annotation_problems.append(f"{run}: Σφάλμα — {str(e)[:30]}")

    if annotation_problems:
        result['annotations_ok'] = False
        result['problems'].extend(annotation_problems)

    # Τελικό αποτέλεσμα
    if result['all_runs_ok'] and result['annotations_ok']:
        result['valid'] = True
        valid_subjects.append(subject)
        print(f"✅ {subject}: OK")
    else:
        result['valid'] = False
        invalid_subjects.append(subject)
        problems_str = " | ".join(result['problems'])
        print(f"❌ {subject}: {problems_str}")

    all_results.append(result)

# ============================================================
# ΒΗΜΑ 4: Σύνοψη
# ============================================================
print("\n" + "=" * 65)
print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ — v2 (Αυστηρός Έλεγχος)")
print("=" * 65)
print(f"\nΣυνολικοί subjects : 109")
print(f"✅ Έγκυροι         : {len(valid_subjects)}")
print(f"❌ Άκυροι          : {len(invalid_subjects)}")

if invalid_subjects:
    print(f"\nΆκυροι subjects:")
    for s in invalid_subjects:
        # Βρίσκουμε τα προβλήματα
        r = next((x for x in all_results if x['subject'] == s), None)
        if r:
            problems = " | ".join(r['problems']) if r['problems'] else "Άγνωστο"
            print(f"  ❌ {s}: {problems}")

# ============================================================
# ΒΗΜΑ 5: Σύγκριση v1 vs v2
# ============================================================
print(f"\n--- ΣΥΓΚΡΙΣΗ v1 vs v2 ---")
print(f"v1 (minimum only) : 108 έγκυροι, 1 άκυρος (S106)")
print(f"v2 (min + max)    : {len(valid_subjects)} έγκυροι, "
      f"{len(invalid_subjects)} άκυροι")

# ============================================================
# ΒΗΜΑ 6: Αποθήκευση
# ============================================================
results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

for r in all_results:
    r['problems'] = " | ".join(r['problems']) if r['problems'] else "Κανένα"

df = pd.DataFrame(all_results)
csv_path = os.path.join(results_dir, "quality_check_v2_results.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV αποθηκεύτηκε: results/quality_check_v2_results.csv")

# Αποθήκευση λίστας έγκυρων subjects
valid_path = os.path.join(results_dir, "valid_subjects_v2.txt")
with open(valid_path, 'w') as f:
    for s in valid_subjects:
        f.write(s + '\n')
print(f"✅ Λίστα έγκυρων  : results/valid_subjects_v2.txt")

print("\n" + "=" * 65)
print(f"Θα χρησιμοποιήσουμε {len(valid_subjects)} subjects στην ανάλυση.")
print("=" * 65)
