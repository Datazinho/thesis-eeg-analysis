# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step3_epochs.py
# Σκοπός  : Τμηματοποίηση (epoching) του EEG σήματος
#           σε μικρά κομμάτια βάσει των annotations T1/T2
# Βήμα    : 3 από 6
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Παίρνουμε το φιλτραρισμένο σήμα και το κόβουμε
# σε μικρά κομμάτια που λέγονται epochs.
#
# ΑΛΛΑΓΗ από παλιά έκδοση:
#   - Χρησιμοποιούμε ΟΛΕΣ τις επαναλήψεις (6 runs ανά συνθήκη)
#   - Imagery : R03, R04, R07, R08, R11, R12
#   - Execution: R05, R06, R09, R10, R13, R14
#   - Τα epochs συγχωνεύονται με mne.concatenate_epochs()
#   - Αποτέλεσμα: ~90 epochs ανά συνθήκη (αντί ~15)
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import mne
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ΒΗΜΑ 1: Ορισμός διαδρομών και παραμέτρων
# ============================================================

DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"

SUBJECT = "S001"

# Runs που χρησιμοποιούμε — ΟΛΕΣ ΟΙ ΕΠΑΝΑΛΗΨΕΙΣ:
# Motor Imagery  : R03, R04, R07, R08, R11, R12
# Motor Execution: R05, R06, R09, R10, R13, R14
# Baseline R01, R02 εξαιρούνται
RUNS_IMAGERY   = ["R03", "R04", "R07", "R08", "R11", "R12"]
RUNS_EXECUTION = ["R05", "R06", "R09", "R10", "R13", "R14"]

# Παράμετροι epoch:
TMIN = -1.0   # 1 δευτερόλεπτο πριν
TMAX =  3.0   # 3 δευτερόλεπτα μετά

print("=" * 60)
print("STEP 3 - Epoching")
print("=" * 60)
print(f"\nSubject        : {SUBJECT}")
print(f"Imagery runs   : {RUNS_IMAGERY}")
print(f"Execution runs : {RUNS_EXECUTION}")
print(f"Παράθυρο epoch : {TMIN} έως {TMAX} δευτερόλεπτα")

# ============================================================
# ΒΗΜΑ 2: Συνάρτηση φόρτωσης και φιλτραρίσματος ΓΙΑ ΕΝΑ RUN
# ============================================================

def load_and_filter(subject, run):
    """
    Φορτώνει ένα EDF αρχείο και το φιλτράρει για mu και beta.

    Παράμετροι:
        subject: π.χ. "S001"
        run    : π.χ. "R03"

    Επιστρέφει:
        raw_mu  : σήμα φιλτραρισμένο 8-13 Hz
        raw_beta: σήμα φιλτραρισμένο 13-30 Hz
    """
    file_path = os.path.join(DATA_PATH, subject, f"{subject}{run}.edf")
    print(f"\n  Φόρτωση: {subject}{run}.edf ...")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw_mu   = raw.copy().filter(l_freq=8.0,  h_freq=13.0, method='fir', verbose=False)
    raw_beta = raw.copy().filter(l_freq=13.0, h_freq=30.0, method='fir', verbose=False)
    print(f"  ✅ Φορτώθηκε και φιλτραρίστηκε!")
    return raw_mu, raw_beta


def get_epochs_from_raw(raw, band_label):
    """
    Δημιουργεί epochs από ένα raw αντικείμενο (T1 και T2 μόνο).
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted = {k: v for k, v in event_id.items() if 'T1' in k or 'T2' in k}
    if not wanted:
        return None
    epochs = mne.Epochs(raw, events, event_id=wanted,
                        tmin=TMIN, tmax=TMAX,
                        baseline=(None, 0), preload=True, verbose=False)
    return epochs


# ============================================================
# ΒΗΜΑ 3: Φόρτωση και συγχώνευση ΟΛΩΝ των Imagery runs
# ============================================================
print("\n--- Φόρτωση Motor IMAGERY (όλα τα runs) ---")

imagery_mu_list   = []
imagery_beta_list = []

for run in RUNS_IMAGERY:
    raw_mu, raw_beta = load_and_filter(SUBJECT, run)
    ep_mu   = get_epochs_from_raw(raw_mu,   'mu')
    ep_beta = get_epochs_from_raw(raw_beta, 'beta')
    if ep_mu is not None and len(ep_mu) > 0:
        imagery_mu_list.append(ep_mu)
    if ep_beta is not None and len(ep_beta) > 0:
        imagery_beta_list.append(ep_beta)

# Συγχώνευση με concatenate_epochs
epochs_imagery_mu   = mne.concatenate_epochs(imagery_mu_list,   verbose=False)
epochs_imagery_beta = mne.concatenate_epochs(imagery_beta_list, verbose=False)
print(f"\n✅ Imagery epochs: {len(epochs_imagery_mu)} (mu), {len(epochs_imagery_beta)} (beta)")

# ============================================================
# ΒΗΜΑ 4: Φόρτωση και συγχώνευση ΟΛΩΝ των Execution runs
# ============================================================
print("\n--- Φόρτωση Motor EXECUTION (όλα τα runs) ---")

execution_mu_list   = []
execution_beta_list = []

for run in RUNS_EXECUTION:
    raw_mu, raw_beta = load_and_filter(SUBJECT, run)
    ep_mu   = get_epochs_from_raw(raw_mu,   'mu')
    ep_beta = get_epochs_from_raw(raw_beta, 'beta')
    if ep_mu is not None and len(ep_mu) > 0:
        execution_mu_list.append(ep_mu)
    if ep_beta is not None and len(ep_beta) > 0:
        execution_beta_list.append(ep_beta)

# Συγχώνευση
epochs_execution_mu   = mne.concatenate_epochs(execution_mu_list,   verbose=False)
epochs_execution_beta = mne.concatenate_epochs(execution_beta_list, verbose=False)
print(f"\n✅ Execution epochs: {len(epochs_execution_mu)} (mu), {len(epochs_execution_beta)} (beta)")

# ============================================================
# ΒΗΜΑ 5: Εκτύπωση αποτελεσμάτων
# ============================================================
print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ EPOCHING ---")
print(f"\nMotor IMAGERY:")
print(f"  epochs_imagery_mu   : {epochs_imagery_mu.get_data().shape}")
print(f"  epochs_imagery_beta : {epochs_imagery_beta.get_data().shape}")
print(f"\nMotor EXECUTION:")
print(f"  epochs_execution_mu   : {epochs_execution_mu.get_data().shape}")
print(f"  epochs_execution_beta : {epochs_execution_beta.get_data().shape}")
print(f"\nΕρμηνεία: (αριθμός_epochs, κανάλια, χρονικά_δείγματα)")

# ============================================================
# ΒΗΜΑ 6: Γράφημα — Σύγκριση Imagery vs Execution
# ============================================================
print("\nΔημιουργία γραφήματος σύγκρισης...")

ch_names = epochs_imagery_mu.ch_names
if 'C3' in ch_names:
    ch_idx = ch_names.index('C3')
    ch_label = 'C3'
elif 'C3.' in ch_names:
    ch_idx = ch_names.index('C3.')
    ch_label = 'C3'
else:
    ch_idx = 0
    ch_label = ch_names[0]

data_imagery_mu   = epochs_imagery_mu.get_data()[:, ch_idx, :]
data_execution_mu = epochs_execution_mu.get_data()[:, ch_idx, :]
mean_imagery   = np.mean(data_imagery_mu,   axis=0)
mean_execution = np.mean(data_execution_mu, axis=0)
sem_imagery    = np.std(data_imagery_mu,    axis=0) / np.sqrt(len(epochs_imagery_mu))
sem_execution  = np.std(data_execution_mu,  axis=0) / np.sqrt(len(epochs_execution_mu))
times = epochs_imagery_mu.times

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Γράφημα 1: Σύγκριση Imagery vs Execution
axes[0].plot(times, mean_imagery   * 1e6, color='steelblue',
             label='Motor Imagery', linewidth=2)
axes[0].fill_between(times,
                     (mean_imagery - sem_imagery) * 1e6,
                     (mean_imagery + sem_imagery) * 1e6,
                     alpha=0.2, color='steelblue')
axes[0].plot(times, mean_execution * 1e6, color='darkorange',
             label='Motor Execution', linewidth=2)
axes[0].fill_between(times,
                     (mean_execution - sem_execution) * 1e6,
                     (mean_execution + sem_execution) * 1e6,
                     alpha=0.2, color='darkorange')
axes[0].axvline(x=0, color='black', linestyle='--',
                linewidth=1, label='Movement onset (t=0)')
axes[0].axvspan(TMIN, 0, alpha=0.05, color='gray', label='Baseline')
axes[0].set_xlabel("Time (s)", fontsize=11)
axes[0].set_ylabel("Amplitude (μV)", fontsize=11)
axes[0].set_title(f"Mean EEG Signal ± SEM\nChannel {ch_label} | Mu Band (8-13 Hz)", fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Γράφημα 2: Αριθμός epochs ανά κατηγορία
categories = ['Motor\nImagery', 'Motor\nExecution']
counts = [len(epochs_imagery_mu), len(epochs_execution_mu)]
colors = ['steelblue', 'darkorange']
bars = axes[1].bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_ylabel("Number of Epochs", fontsize=11)
axes[1].set_title("Epochs per Condition", fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

fig.suptitle(f"EEG Epoch Comparison — {SUBJECT} | Channel {ch_label} | Mu Band (8-13 Hz)",
             fontsize=13, fontweight='bold')
plt.tight_layout()

results_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "step3_epochs_S001.png"
)
plt.savefig(results_path, dpi=150, bbox_inches='tight')
print(f"✅ Γράφημα αποθηκεύτηκε: results/step3_epochs_S001.png")
plt.show()

# ============================================================
# ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ
# ============================================================
print("\n--- ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ ---")
all_ok = all(len(e) > 0 for e in [
    epochs_imagery_mu, epochs_imagery_beta,
    epochs_execution_mu, epochs_execution_beta
])

if all_ok:
    print("✅ ΟΛΑ ΣΩΣΤΑ! Τα 4 σύνολα epochs δημιουργήθηκαν επιτυχώς.")
    print(f"   Imagery MU   : {len(epochs_imagery_mu)} epochs")
    print(f"   Imagery BETA : {len(epochs_imagery_beta)} epochs")
    print(f"   Execution MU : {len(epochs_execution_mu)} epochs")
    print(f"   Execution BETA: {len(epochs_execution_beta)} epochs")
else:
    print("⚠️  ΠΡΟΒΛΗΜΑ: Κάποια σύνολα epochs είναι άδεια!")

print("\n" + "=" * 60)
print("Επόμενο βήμα: step4_connectivity.py")
print("=" * 60)
