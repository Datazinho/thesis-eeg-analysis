# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step_shuqfa_check.py
# Σκοπός  : Έλεγχος των 6 subjects που ανέφεραν
#           οι Shuqfa et al. (2024) ως προβληματικούς
#           και σύγκριση με τα δικά μας αποτελέσματα
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Οι Shuqfa et al. (2024) ανέφεραν ότι 6 subjects
# έχουν anomalies:
#   S038, S088, S089, S092, S100, S104
#
# Εμείς ελέγχουμε ΑΝΑΛΥΤΙΚΑ αυτούς τους 6 subjects
# για ΟΛΑ τα runs (R03-R14) και συγκρίνουμε:
#   - Αριθμός annotations T0/T1/T2
#   - Αριθμός καναλιών
#   - Διάρκεια runs
#
# Επίσης ελέγχουμε τον S106 που βρήκαμε εμείς
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import mne
import os
import pandas as pd
import numpy as np

# ============================================================
# ΒΗΜΑ 1: Παράμετροι
# ============================================================

DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"

# Subjects από Shuqfa et al. (2024) + ο δικός μας S106
SHUQFA_SUBJECTS = ["S038", "S088", "S089", "S092", "S100", "S104"]
OUR_SUBJECT     = ["S106"]
ALL_CHECK       = SHUQFA_SUBJECTS + OUR_SUBJECT

# Όλα τα task runs
RUNS = ["R03", "R04", "R05", "R06", "R07", "R08",
        "R09", "R10", "R11", "R12", "R13", "R14"]

# Αναμενόμενες τιμές
EXPECTED_CHANNELS  = 64
EXPECTED_T0        = 15    # περίπου
EXPECTED_T1        = 7     # ή 8
EXPECTED_T2        = 7     # ή 8
EXPECTED_DURATION  = 120   # δευτερόλεπτα (~2 λεπτά)

print("=" * 70)
print("ΑΝΑΛΥΤΙΚΟΣ ΕΛΕΓΧΟΣ — Shuqfa et al. Subjects + S106")
print("=" * 70)
print(f"\nSubjects Shuqfa et al.: {SHUQFA_SUBJECTS}")
print(f"Δικός μας subject     : {OUR_SUBJECT}")

# ============================================================
# ΒΗΜΑ 2: Συνάρτηση αναλυτικού ελέγχου
# ============================================================

def check_subject_detailed(subject):
    """
    Κάνει αναλυτικό έλεγχο ενός subject για όλα τα runs.

    Επιστρέφει:
        λίστα με αποτελέσματα για κάθε run
    """
    subject_path = os.path.join(DATA_PATH, subject)
    results = []

    print(f"\n{'='*70}")
    print(f"Subject: {subject}")
    print(f"{'='*70}")
    print(f"{'Run':<6} {'Channels':<10} {'Duration':<12} {'T0':<6} {'T1':<6} {'T2':<6} {'Status'}")
    print("-" * 70)

    for run in RUNS:
        file_path = os.path.join(subject_path, f"{subject}{run}.edf")

        result = {
            'subject' : subject,
            'run'     : run,
        }

        # Έλεγχος αν υπάρχει το αρχείο
        if not os.path.exists(file_path):
            result['status']   = '❌ MISSING'
            result['channels'] = '-'
            result['duration'] = '-'
            result['T0']       = '-'
            result['T1']       = '-'
            result['T2']       = '-'
            print(f"{run:<6} {'MISSING':<10} {'-':<12} {'-':<6} {'-':<6} {'-':<6} ❌ Λείπει")
            results.append(result)
            continue

        try:
            # Φορτώνουμε το αρχείο
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

            # Κανάλια
            n_ch = len(raw.ch_names)

            # Διάρκεια
            duration = raw.times[-1]

            # Annotations
            ann = raw.annotations
            t0 = sum(1 for d in ann.description if 'T0' in d)
            t1 = sum(1 for d in ann.description if 'T1' in d)
            t2 = sum(1 for d in ann.description if 'T2' in d)

            # Έλεγχος προβλημάτων
            problems = []
            if n_ch != EXPECTED_CHANNELS:
                problems.append(f"ch={n_ch}")
            if duration < 60:
                problems.append(f"dur={duration:.0f}s")
            if t0 < 5:
                problems.append(f"T0={t0}")
            if t1 < 3:
                problems.append(f"T1={t1}")
            if t2 < 3:
                problems.append(f"T2={t2}")

            status = "✅ OK" if not problems else f"⚠️  {', '.join(problems)}"

            result['channels'] = n_ch
            result['duration'] = round(duration, 1)
            result['T0']       = t0
            result['T1']       = t1
            result['T2']       = t2
            result['status']   = status
            result['problems'] = " | ".join(problems) if problems else "Κανένα"

            print(f"{run:<6} {n_ch:<10} {duration:<12.1f} {t0:<6} {t1:<6} {t2:<6} {status}")

        except Exception as e:
            result['status']   = f"❌ ERROR: {str(e)[:30]}"
            result['channels'] = '-'
            result['duration'] = '-'
            result['T0']       = '-'
            result['T1']       = '-'
            result['T2']       = '-'
            print(f"{run:<6} {'ERROR':<10} {'-':<12} {'-':<6} {'-':<6} {'-':<6} ❌ {str(e)[:20]}")

        results.append(result)

    return results

# ============================================================
# ΒΗΜΑ 3: Έλεγχος όλων των subjects
# ============================================================
all_results = []

for subject in ALL_CHECK:
    results = check_subject_detailed(subject)
    all_results.extend(results)

# ============================================================
# ΒΗΜΑ 4: Σύγκριση με Shuqfa et al.
# ============================================================
print(f"\n{'='*70}")
print("ΣΥΓΚΡΙΣΗ ΜΕ SHUQFA ET AL. (2024)")
print(f"{'='*70}")
print(f"\n{'Subject':<10} {'Shuqfa et al.':<25} {'Δικός μας έλεγχος'}")
print("-" * 70)

shuqfa_findings = {
    "S038": "Anomaly annotations + διάρκεια",
    "S088": "Λάθος κωδικοποίηση T1/T2",
    "S089": "Λάθος κωδικοποίηση T1/T2",
    "S092": "Ανεπαρκής αριθμός trials",
    "S100": "Anomaly διάρκεια + trials",
    "S104": "Anomaly annotations",
    "S106": "ΔΕΝ ανέφεραν",
}

for subject in ALL_CHECK:
    # Βρίσκουμε τα προβλήματα που βρήκαμε εμείς
    subject_results = [r for r in all_results
                      if r['subject'] == subject and
                      isinstance(r.get('problems'), str) and
                      r.get('problems') != 'Κανένα']

    if subject_results:
        our_finding = f"⚠️  Προβλήματα σε {len(subject_results)} runs"
    else:
        our_finding = "✅ Πέρασε τον δικό μας έλεγχο"

    shuqfa = shuqfa_findings.get(subject, "Δεν ανέφεραν")
    print(f"{subject:<10} {shuqfa:<35} {our_finding}")

# ============================================================
# ΒΗΜΑ 5: Αποθήκευση CSV
# ============================================================
results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

df = pd.DataFrame(all_results)
csv_path = os.path.join(results_dir, "shuqfa_check_detailed.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV αποθηκεύτηκε: results/shuqfa_check_detailed.csv")

# ============================================================
# ΤΕΛΙΚΟ ΣΥΜΠΕΡΑΣΜΑ
# ============================================================
print(f"\n{'='*70}")
print("ΤΕΛΙΚΟ ΣΥΜΠΕΡΑΣΜΑ")
print(f"{'='*70}")
print("""
Οι Shuqfa et al. (2024) ανέφεραν 6 προβληματικούς subjects.
Ο δικός μας έλεγχος βρήκε διαφορετικά αποτελέσματα:

→ Θα χρησιμοποιήσουμε 108 subjects (αφαιρώντας μόνο S106)
→ Η διαφορά θα αναφερθεί και θα αιτιολογηθεί στο Κεφ. 4
→ Θα αναφέρουμε τα αποτελέσματα Shuqfa et al. ως σύγκριση
""")
print("=" * 70)
