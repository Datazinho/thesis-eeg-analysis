# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : loop_all_subjects_v2.py
# Σκοπός  : Τρέχει το pipeline για ΟΛΟΥΣ τους 105 έγκυρους
#           subjects χρησιμοποιώντας ΟΛΕΣ τις επαναλήψεις
#           (R03,R04,R07,R08,R11,R12 για Imagery και
#            R05,R06,R09,R10,R13,R14 για Execution)
# ============================================================
# ΑΛΛΑΓΕΣ από v1:
#   - Χρησιμοποιούμε ΟΛΕΣ τις επαναλήψεις (6 runs ανά συνθήκη)
#   - Τα epochs συγχωνεύονται από όλα τα runs πριν τον
#     υπολογισμό connectivity
#   - Αναμενόμενος χρόνος: 2-4 ώρες
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import mne
import os
import numpy as np
import networkx as nx
import pandas as pd
import time

try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    print("⚠️  python-louvain δεν βρέθηκε — Modularity δεν θα υπολογιστεί")

# ============================================================
# ΒΗΜΑ 1: Παράμετροι
# ============================================================

DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

# ============================================================
# ΟΛΕΣ ΟΙ ΕΠΑΝΑΛΗΨΕΙΣ ΑΝΑ ΣΥΝΘΗΚΗ
# ============================================================
RUNS_IMAGERY   = ["R03", "R04", "R07", "R08", "R11", "R12"]
RUNS_EXECUTION = ["R05", "R06", "R09", "R10", "R13", "R14"]

# Epoching
TMIN = -1.0
TMAX =  3.0

# Threshold connectivity
THRESHOLD = 0.5

# Ονόματα ηλεκτροδίων
ELECTRODE_NAMES = [
    'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
    'Fp1', 'Fpz', 'Fp2',
    'Af7', 'Af3', 'Afz', 'Af4', 'Af8',
    'F7',  'F5',  'F3',  'F1',  'Fz',  'F2',  'F4',  'F6',  'F8',
    'Ft7', 'Ft8',
    'T7',  'T8',  'T9',  'T10',
    'Tp7', 'Tp8',
    'P7',  'P5',  'P3',  'P1',  'Pz',  'P2',  'P4',  'P6',  'P8',
    'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
    'O1',  'Oz',  'O2',  'Iz'
]

# ============================================================
# ΒΗΜΑ 2: Φόρτωση λίστας έγκυρων subjects
# ============================================================
valid_subjects_path = os.path.join(RESULTS_DIR, "valid_subjects_v2.txt")

with open(valid_subjects_path, 'r') as f:
    valid_subjects = [line.strip() for line in f.readlines()]

print("=" * 65)
print("LOOP v2 — Pipeline για όλους τους έγκυρους subjects")
print("         με ΟΛΕΣ τις επαναλήψεις")
print("=" * 65)
print(f"\nΈγκυροι subjects : {len(valid_subjects)}")
print(f"Imagery runs     : {RUNS_IMAGERY}")
print(f"Execution runs   : {RUNS_EXECUTION}")
print(f"Threshold        : r > {THRESHOLD}")
print(f"\n⚠️  Αυτό θα πάρει 2-4 ώρες...")
print("-" * 65)

# ============================================================
# ΒΗΜΑ 3: Βοηθητικές συναρτήσεις
# ============================================================

def get_epochs_single_run(subject, run, band):
    """Φορτώνει, φιλτράρει και κάνει epoching ΓΙΑ ΕΝΑ run."""
    file_path = os.path.join(DATA_PATH, subject, f"{subject}{run}.edf")

    # Έλεγχος αν υπάρχει το αρχείο
    if not os.path.exists(file_path):
        return None

    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        if band == 'mu':
            raw.filter(l_freq=8.0,  h_freq=13.0, method='fir', verbose=False)
        else:
            raw.filter(l_freq=13.0, h_freq=30.0, method='fir', verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        wanted = {k: v for k, v in event_id.items() if 'T1' in k or 'T2' in k}

        if not wanted:
            return None

        epochs = mne.Epochs(raw, events, event_id=wanted,
                            tmin=TMIN, tmax=TMAX,
                            baseline=(None, 0), preload=True, verbose=False)

        if len(epochs) == 0:
            return None

        return epochs

    except Exception:
        return None


def get_epochs_all_runs(subject, runs, band):
    """
    Φορτώνει και συγχωνεύει epochs από ΟΛΕΣ τις επαναλήψεις.
    Επιστρέφει συγχωνευμένα epochs ή None αν αποτύχουν όλα.
    """
    all_epochs = []

    for run in runs:
        ep = get_epochs_single_run(subject, run, band)
        if ep is not None:
            all_epochs.append(ep)

    if len(all_epochs) == 0:
        return None

    # Συγχώνευση όλων των epochs
    if len(all_epochs) == 1:
        return all_epochs[0]

    return mne.concatenate_epochs(all_epochs, verbose=False)


def compute_connectivity(epochs):
    """Υπολογίζει μέσο πίνακα Pearson correlation 64x64."""
    data = epochs.get_data()
    n_epochs = data.shape[0]
    all_corr = np.zeros((n_epochs, 64, 64))

    for i in range(n_epochs):
        corr = np.corrcoef(data[i])
        all_corr[i] = np.abs(corr)

    conn_mean = np.mean(all_corr, axis=0)
    conn_bin  = (conn_mean > THRESHOLD).astype(float)
    np.fill_diagonal(conn_bin,  0)
    np.fill_diagonal(conn_mean, 0)
    return conn_bin


def build_graph(adj_matrix):
    """Κατασκευάζει γράφο NetworkX από δυαδικό πίνακα."""
    G = nx.Graph()
    for i in range(64):
        G.add_node(i, label=ELECTRODE_NAMES[i])
    for i in range(64):
        for j in range(i + 1, 64):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G


def compute_metrics(G, subject, condition, band, n_epochs):
    """Υπολογίζει όλες τις μετρικές για έναν γράφο."""
    metrics = {
        'subject'  : subject,
        'condition': condition,
        'band'     : band,
        'n_epochs' : n_epochs,  # ΝΕΟΝ: καταγράφουμε πόσα epochs χρησιμοποιήθηκαν
    }

    # Clustering Coefficient
    metrics['clustering_coefficient'] = round(nx.average_clustering(G), 4)

    # Global Efficiency
    metrics['global_efficiency'] = round(nx.global_efficiency(G), 4)

    # Modularity
    if HAS_COMMUNITY:
        try:
            np.random.seed(42)  # Αναπαραγωγιμότητα Louvain
            partition  = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
            metrics['modularity']    = round(modularity, 4)
            metrics['n_communities'] = len(set(partition.values()))
        except:
            metrics['modularity']    = None
            metrics['n_communities'] = None
    else:
        metrics['modularity']    = None
        metrics['n_communities'] = None

    # Betweenness Centrality
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    bc_values = list(bc_dict.values())
    metrics['bc_mean'] = round(np.mean(bc_values), 4)
    metrics['bc_max']  = round(np.max(bc_values),  4)
    max_node = max(bc_dict, key=bc_dict.get)
    metrics['bc_max_electrode'] = ELECTRODE_NAMES[max_node]

    # Density
    n_edges = G.number_of_edges()
    total   = 64 * 63 / 2
    metrics['density'] = round(n_edges / total * 100, 2)
    metrics['n_edges'] = n_edges

    return metrics

# ============================================================
# ΒΗΜΑ 4: Κύριος loop
# ============================================================
all_metrics  = []
failed       = []
start_time   = time.time()

for idx, subject in enumerate(valid_subjects):
    subject_start = time.time()

    print(f"\n[{idx+1}/{len(valid_subjects)}] {subject} ...", end=" ")

    try:
        # Φόρτωση και συγχώνευση epochs από ΟΛΕΣ τις επαναλήψεις
        ep_img_mu   = get_epochs_all_runs(subject, RUNS_IMAGERY,   'mu')
        ep_img_beta = get_epochs_all_runs(subject, RUNS_IMAGERY,   'beta')
        ep_exc_mu   = get_epochs_all_runs(subject, RUNS_EXECUTION, 'mu')
        ep_exc_beta = get_epochs_all_runs(subject, RUNS_EXECUTION, 'beta')

        # Έλεγχος αν τα epochs είναι έγκυρα
        if any(e is None or len(e) == 0
               for e in [ep_img_mu, ep_img_beta, ep_exc_mu, ep_exc_beta]):
            print("⚠️  Παράλειψη — άδεια epochs")
            failed.append(subject)
            continue

        # Καταγραφή αριθμού epochs
        n_img = len(ep_img_mu)
        n_exc = len(ep_exc_mu)

        # Connectivity matrices
        conn_img_mu   = compute_connectivity(ep_img_mu)
        conn_img_beta = compute_connectivity(ep_img_beta)
        conn_exc_mu   = compute_connectivity(ep_exc_mu)
        conn_exc_beta = compute_connectivity(ep_exc_beta)

        # Γράφοι
        G_img_mu   = build_graph(conn_img_mu)
        G_img_beta = build_graph(conn_img_beta)
        G_exc_mu   = build_graph(conn_exc_mu)
        G_exc_beta = build_graph(conn_exc_beta)

        # Μετρικές
        all_metrics.append(compute_metrics(G_img_mu,   subject, 'imagery',   'mu',   n_img))
        all_metrics.append(compute_metrics(G_img_beta, subject, 'imagery',   'beta', n_img))
        all_metrics.append(compute_metrics(G_exc_mu,   subject, 'execution', 'mu',   n_exc))
        all_metrics.append(compute_metrics(G_exc_beta, subject, 'execution', 'beta', n_exc))

        elapsed = time.time() - subject_start
        print(f"✅ (img:{n_img} epochs, exc:{n_exc} epochs, {elapsed:.1f}s)")

        # Αποθήκευση ανά 10 subjects (για ασφάλεια)
        if (idx + 1) % 10 == 0:
            df_temp = pd.DataFrame(all_metrics)
            temp_path = os.path.join(RESULTS_DIR, "all_metrics_temp_v2.csv")
            df_temp.to_csv(temp_path, index=False, encoding='utf-8-sig')
            print(f"   💾 Προσωρινή αποθήκευση: {idx+1} subjects")

    except Exception as e:
        print(f"❌ Σφάλμα: {str(e)[:50]}")
        failed.append(subject)

# ============================================================
# ΒΗΜΑ 5: Αποθήκευση τελικών αποτελεσμάτων
# ============================================================
print("\n" + "=" * 65)
print("ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
print("=" * 65)

df = pd.DataFrame(all_metrics)
csv_path = os.path.join(RESULTS_DIR, "all_metrics_final_v2.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ CSV αποθηκεύτηκε: results/all_metrics_final_v2.csv")
print(f"   Γραμμές: {len(df)} ({len(valid_subjects)} subjects × 4 κατηγορίες)")

# ============================================================
# ΒΗΜΑ 6: Σύνοψη
# ============================================================
total_time = time.time() - start_time
print(f"\n--- ΣΥΝΟΨΗ ---")
print(f"Συνολικός χρόνος   : {total_time/60:.1f} λεπτά")
print(f"Επιτυχής subjects  : {len(valid_subjects) - len(failed)}")
print(f"Αποτυχημένοι       : {len(failed)}")

if failed:
    print(f"\nΑποτυχημένοι subjects: {failed}")

# Γρήγορη στατιστική
print(f"\n--- ΓΡΗΓΟΡΗ ΣΤΑΤΙΣΤΙΚΗ ---")
for condition in ['imagery', 'execution']:
    for band in ['mu', 'beta']:
        subset = df[(df['condition']==condition) & (df['band']==band)]
        cc_mean = subset['clustering_coefficient'].mean()
        ge_mean = subset['global_efficiency'].mean()
        n_ep    = subset['n_epochs'].mean()
        print(f"{condition} {band}: CC={cc_mean:.3f}, GE={ge_mean:.3f}, μέσο epochs={n_ep:.0f}")

print("\n" + "=" * 65)
print("🎉 Loop v2 ολοκληρώθηκε!")
print("Επόμενο βήμα: statistical_analysis.py")
print("  (χρησιμοποίησε all_metrics_final_v2.csv)")
print("=" * 65)
