# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step4_connectivity.py
# Σκοπός  : Υπολογισμός πίνακα λειτουργικής συνδεσιμότητας
#           64×64 με Pearson Correlation
# Βήμα    : 4 από 6
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Για κάθε epoch υπολογίζουμε πόσο "συνδεδεμένα" είναι
# τα 64 ηλεκτρόδια μεταξύ τους.
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
import matplotlib.gridspec as gridspec

# ============================================================
# ΒΗΜΑ 1: Παράμετροι
# ============================================================

DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"
SUBJECT   = "S001"

# Runs που χρησιμοποιούμε — ΟΛΕΣ ΟΙ ΕΠΑΝΑΛΗΨΕΙΣ:
RUNS_IMAGERY   = ["R03", "R04", "R07", "R08", "R11", "R12"]
RUNS_EXECUTION = ["R05", "R06", "R09", "R10", "R13", "R14"]

TMIN = -1.0
TMAX =  3.0
THRESHOLD = 0.5

print("=" * 60)
print("STEP 4 - Connectivity Matrix 64×64")
print("=" * 60)
print(f"\nSubject        : {SUBJECT}")
print(f"Imagery runs   : {RUNS_IMAGERY}")
print(f"Execution runs : {RUNS_EXECUTION}")
print(f"Threshold      : r > {THRESHOLD}")

# ============================================================
# ΒΗΜΑ 2: Συνάρτηση φόρτωσης για ΕΝΑ run
# ============================================================

def get_epochs_single_run(subject, run, band='mu'):
    """
    Φορτώνει, φιλτράρει και κάνει epoching ΓΙΑ ΕΝΑ run.

    Παράμετροι:
        subject: π.χ. "S001"
        run    : π.χ. "R03"
        band   : 'mu' (8-13 Hz) ή 'beta' (13-30 Hz)

    Επιστρέφει:
        epochs: αντικείμενο MNE Epochs ή None αν αποτύχει
    """
    file_path = os.path.join(DATA_PATH, subject, f"{subject}{run}.edf")
    if not os.path.exists(file_path):
        return None
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        if band == 'mu':
            raw.filter(l_freq=8.0, h_freq=13.0, method='fir', verbose=False)
        else:
            raw.filter(l_freq=13.0, h_freq=30.0, method='fir', verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        wanted = {k: v for k, v in event_id.items() if 'T1' in k or 'T2' in k}
        if not wanted:
            return None
        epochs = mne.Epochs(raw, events, event_id=wanted,
                            tmin=TMIN, tmax=TMAX,
                            baseline=(None, 0), preload=True, verbose=False)
        return epochs if len(epochs) > 0 else None
    except Exception:
        return None


def get_epochs_all_runs(subject, runs, band='mu'):
    """
    Φορτώνει και συγχωνεύει epochs από ΟΛΕΣ τις επαναλήψεις.
    """
    all_epochs = []
    for run in runs:
        ep = get_epochs_single_run(subject, run, band)
        if ep is not None:
            all_epochs.append(ep)
    if len(all_epochs) == 0:
        return None
    if len(all_epochs) == 1:
        return all_epochs[0]
    return mne.concatenate_epochs(all_epochs, verbose=False)

# ============================================================
# ΒΗΜΑ 3: Φόρτωση όλων των epochs
# ============================================================
print("\nΦόρτωση epochs από όλα τα runs...")

print("  imagery_mu   ...", end=" ")
epochs_imagery_mu   = get_epochs_all_runs(SUBJECT, RUNS_IMAGERY,   'mu')
print(f"✅ {len(epochs_imagery_mu)} epochs")

print("  imagery_beta ...", end=" ")
epochs_imagery_beta = get_epochs_all_runs(SUBJECT, RUNS_IMAGERY,   'beta')
print(f"✅ {len(epochs_imagery_beta)} epochs")

print("  execution_mu ...", end=" ")
epochs_execution_mu  = get_epochs_all_runs(SUBJECT, RUNS_EXECUTION, 'mu')
print(f"✅ {len(epochs_execution_mu)} epochs")

print("  execution_beta...", end=" ")
epochs_execution_beta = get_epochs_all_runs(SUBJECT, RUNS_EXECUTION, 'beta')
print(f"✅ {len(epochs_execution_beta)} epochs")

# ============================================================
# ΒΗΜΑ 4: Συνάρτηση υπολογισμού Connectivity Matrix
# ============================================================

def compute_connectivity(epochs):
    """
    Υπολογίζει τον μέσο πίνακα Pearson correlation 64×64
    πάνω από όλα τα epochs.
    """
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    all_corr = np.zeros((n_epochs, n_channels, n_channels))
    for i in range(n_epochs):
        corr_matrix = np.corrcoef(data[i])
        all_corr[i] = np.abs(corr_matrix)
    conn_mean = np.mean(all_corr, axis=0)
    conn_bin  = (conn_mean > THRESHOLD).astype(float)
    np.fill_diagonal(conn_bin,  0)
    np.fill_diagonal(conn_mean, 0)
    return conn_mean, conn_bin

# ============================================================
# ΒΗΜΑ 5: Υπολογισμός Connectivity για όλες τις κατηγορίες
# ============================================================
print("\nΥπολογισμός Connectivity Matrices...")

print("  imagery_mu   ...", end=" ")
conn_imagery_mu_mean,   conn_imagery_mu_bin   = compute_connectivity(epochs_imagery_mu)
print("✅")

print("  imagery_beta ...", end=" ")
conn_imagery_beta_mean, conn_imagery_beta_bin = compute_connectivity(epochs_imagery_beta)
print("✅")

print("  execution_mu ...", end=" ")
conn_exec_mu_mean,   conn_exec_mu_bin   = compute_connectivity(epochs_execution_mu)
print("✅")

print("  execution_beta...", end=" ")
conn_exec_beta_mean, conn_exec_beta_bin = compute_connectivity(epochs_execution_beta)
print("✅")

# ============================================================
# ΒΗΜΑ 6: Εκτύπωση στατιστικών
# ============================================================
def get_density(bin_matrix):
    n = bin_matrix.shape[0]
    total  = n * (n - 1)
    actual = np.sum(bin_matrix)
    return (actual / total) * 100

print("\n--- ΣΤΑΤΙΣΤΙΚΑ CONNECTIVITY ---")
print(f"{'Κατηγορία':<25} {'Μέση r':<12} {'Density':<12} {'Συνδέσεις'}")
print("-" * 60)
for name, mean_m, bin_m in [
    ("Imagery MU",      conn_imagery_mu_mean,   conn_imagery_mu_bin),
    ("Imagery BETA",    conn_imagery_beta_mean, conn_imagery_beta_bin),
    ("Execution MU",    conn_exec_mu_mean,      conn_exec_mu_bin),
    ("Execution BETA",  conn_exec_beta_mean,    conn_exec_beta_bin),
]:
    mean_r  = np.mean(mean_m[mean_m > 0])
    density = get_density(bin_m)
    n_edges = int(np.sum(bin_m) / 2)
    print(f"{name:<25} {mean_r:<12.3f} {density:<11.1f}% {n_edges}")

# ============================================================
# ΒΗΜΑ 7: Γράφημα — 4 Heatmaps
# ============================================================
print("\nΔημιουργία heatmaps...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle(f"Functional Connectivity Matrices 64×64 — {SUBJECT}\n"
             f"Pearson Correlation (Continuous Values)",
             fontsize=14, fontweight='bold')

plots = [
    (conn_imagery_mu_mean,   f"Motor Imagery — Mu Band\n(8-13 Hz)",    "Blues"),
    (conn_imagery_beta_mean, f"Motor Imagery — Beta Band\n(13-30 Hz)", "Blues"),
    (conn_exec_mu_mean,      f"Motor Execution — Mu Band\n(8-13 Hz)",  "Reds"),
    (conn_exec_beta_mean,    f"Motor Execution — Beta Band\n(13-30 Hz)","Reds"),
]

for idx, (matrix, title, cmap) in enumerate(plots):
    ax = fig.add_subplot(2, 2, idx + 1)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Electrode index", fontsize=9)
    ax.set_ylabel("Electrode index", fontsize=9)
    mean_r = np.mean(matrix[matrix > 0])
    ax.text(0.02, 0.98, f"Mean r = {mean_r:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()

results_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "step4_connectivity_S001.png"
)
plt.savefig(results_path, dpi=150, bbox_inches='tight')
print(f"✅ Γράφημα αποθηκεύτηκε: results/step4_connectivity_S001.png")
plt.show()

# ============================================================
# ΒΗΜΑ 8: Αποθήκευση πινάκων για step5 και step6
# ============================================================
results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

np.save(os.path.join(results_dir, "conn_imagery_mu_bin.npy"),   conn_imagery_mu_bin)
np.save(os.path.join(results_dir, "conn_imagery_beta_bin.npy"), conn_imagery_beta_bin)
np.save(os.path.join(results_dir, "conn_exec_mu_bin.npy"),      conn_exec_mu_bin)
np.save(os.path.join(results_dir, "conn_exec_beta_bin.npy"),    conn_exec_beta_bin)
print("✅ Πίνακες αποθηκεύτηκαν στο results/ (.npy αρχεία)")

# ============================================================
# ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ
# ============================================================
print("\n--- ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ ---")
print(f"✅ Connectivity matrices υπολογίστηκαν!")
print(f"   Διαστάσεις κάθε πίνακα: {conn_imagery_mu_bin.shape}")
print(f"   Threshold: r > {THRESHOLD}")
print(f"   Imagery epochs  : {len(epochs_imagery_mu)}")
print(f"   Execution epochs: {len(epochs_execution_mu)}")

print("\n" + "=" * 60)
print("Επόμενο βήμα: step5_graph.py")
print("=" * 60)
