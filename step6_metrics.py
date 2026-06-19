# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step6_metrics.py
# Σκοπός  : Υπολογισμός μετρικών θεωρίας γράφων
# Βήμα    : 6 από 6 (ΤΕΛΕΥΤΑΙΟ!)
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Για κάθε γράφο υπολογίζουμε 4 μετρικές:
#
# 1. Clustering Coefficient
#    → Πόσο οι γειτονικοί κόμβοι συνδέονται μεταξύ τους
#    → Μετράει "τοπική οργάνωση" του δικτύου
#    → Υψηλή τιμή = το δίκτυο έχει "ομάδες" κόμβων
#
# 2. Global Efficiency
#    → Πόσο αποδοτικά ανταλλάσσεται πληροφορία
#    → Μετράει "παγκόσμια επικοινωνία" του δικτύου
#    → Υψηλή τιμή = γρήγορη επικοινωνία μεταξύ όλων
#
# 3. Modularity
#    → Αν το δίκτυο χωρίζεται σε ξεχωριστές κοινότητες
#    → Υψηλή τιμή = ξεκάθαρος διαχωρισμός σε modules
#
# 4. Betweenness Centrality
#    → Ποια ηλεκτρόδια είναι "γέφυρες" επικοινωνίας
#    → 1 τιμή ανά κόμβο (64 τιμές συνολικά)
#
# Αποτέλεσμα: CSV αρχείο με όλες τις μετρικές
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# community detection — για Modularity
# Εγκατάσταση αν δεν υπάρχει: pip install python-louvain
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    print("⚠️  Η βιβλιοθήκη 'community' δεν βρέθηκε.")
    print("   Τρέξε: pip install python-louvain")
    print("   Συνεχίζουμε χωρίς Modularity...")

# ============================================================
# ΒΗΜΑ 1: Φόρτωση πινάκων από step4
# ============================================================
results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

print("=" * 60)
print("STEP 6 - Υπολογισμός Μετρικών Γράφων")
print("=" * 60)

print("\nΦόρτωση πινάκων connectivity...")
conn_imagery_mu_bin   = np.load(os.path.join(results_dir, "conn_imagery_mu_bin.npy"))
conn_imagery_beta_bin = np.load(os.path.join(results_dir, "conn_imagery_beta_bin.npy"))
conn_exec_mu_bin      = np.load(os.path.join(results_dir, "conn_exec_mu_bin.npy"))
conn_exec_beta_bin    = np.load(os.path.join(results_dir, "conn_exec_beta_bin.npy"))
print("✅ Πίνακες φορτώθηκαν!")

# ============================================================
# ΒΗΜΑ 2: Ονόματα ηλεκτροδίων
# ============================================================
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
# ΒΗΜΑ 3: Συνάρτηση κατασκευής γράφου
# ============================================================
def build_graph(adj_matrix, electrode_names):
    G = nx.Graph()
    for i, name in enumerate(electrode_names):
        G.add_node(i, label=name)
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G

# ============================================================
# ΒΗΜΑ 4: Κατασκευή γράφων
# ============================================================
print("\nΚατασκευή γράφων...")
G_imagery_mu   = build_graph(conn_imagery_mu_bin,   ELECTRODE_NAMES)
G_imagery_beta = build_graph(conn_imagery_beta_bin, ELECTRODE_NAMES)
G_exec_mu      = build_graph(conn_exec_mu_bin,      ELECTRODE_NAMES)
G_exec_beta    = build_graph(conn_exec_beta_bin,    ELECTRODE_NAMES)
print("✅ Γράφοι έτοιμοι!")

# ============================================================
# ΒΗΜΑ 5: Συνάρτηση υπολογισμού μετρικών
# ============================================================
def compute_metrics(G, condition, band, subject="S001"):
    """
    Υπολογίζει όλες τις μετρικές για έναν γράφο.

    Παράμετροι:
        G        : γράφος NetworkX
        condition: 'imagery' ή 'execution'
        band     : 'mu' ή 'beta'
        subject  : π.χ. 'S001'

    Επιστρέφει:
        dict με όλες τις μετρικές
    """
    metrics = {
        'subject'  : subject,
        'condition': condition,
        'band'     : band,
    }

    # --- 1. Clustering Coefficient ---
    # nx.average_clustering(): μέσος όρος clustering coefficient
    # για όλους τους κόμβους
    # Τιμή 0-1: 1 = όλοι οι γείτονες συνδέονται μεταξύ τους
    cc = nx.average_clustering(G)
    metrics['clustering_coefficient'] = round(cc, 4)
    print(f"   Clustering Coefficient : {cc:.4f}")

    # --- 2. Global Efficiency ---
    # nx.global_efficiency(): μέτρο αποδοτικότητας παγκόσμιας
    # επικοινωνίας — βασίζεται στο αντίστροφο των συντομότερων
    # μονοπατιών μεταξύ όλων των ζευγών κόμβων
    # Τιμή 0-1: 1 = τέλεια αποδοτικότητα
    ge = nx.global_efficiency(G)
    metrics['global_efficiency'] = round(ge, 4)
    print(f"   Global Efficiency      : {ge:.4f}")

    # --- 3. Modularity ---
    # Χρησιμοποιούμε αλγόριθμο Louvain για community detection
    # Βρίσκει ποιοι κόμβοι ανήκουν στην ίδια "κοινότητα"
    # Modularity Q: -0.5 έως 1
    # Q > 0.3 = καλός διαχωρισμός σε κοινότητες
    if HAS_COMMUNITY:
        np.random.seed(42)  # Αναπαραγωγιμότητα Louvain
        partition   = community_louvain.best_partition(G)
        modularity  = community_louvain.modularity(partition, G)
        n_communities = len(set(partition.values()))
        metrics['modularity']     = round(modularity, 4)
        metrics['n_communities']  = n_communities
        print(f"   Modularity             : {modularity:.4f}")
        print(f"   Αριθμός communities    : {n_communities}")
    else:
        metrics['modularity']    = None
        metrics['n_communities'] = None
        print("   Modularity             : N/A (χρειάζεται python-louvain)")

    # --- 4. Betweenness Centrality ---
    # nx.betweenness_centrality(): για κάθε κόμβο, πόσα
    # συντομότερα μονοπάτια περνούν από αυτόν
    # Υψηλή τιμή = ο κόμβος είναι "γέφυρα" επικοινωνίας
    bc_dict = nx.betweenness_centrality(G, normalized=True)

    # Αποθηκεύουμε για κάθε ηλεκτρόδιο ξεχωριστά
    for i, name in enumerate(ELECTRODE_NAMES):
        metrics[f'bc_{name}'] = round(bc_dict.get(i, 0), 4)

    # Επίσης αποθηκεύουμε μέσο όρο και max
    bc_values = list(bc_dict.values())
    metrics['bc_mean'] = round(np.mean(bc_values), 4)
    metrics['bc_max']  = round(np.max(bc_values), 4)

    # Βρίσκουμε ποιο ηλεκτρόδιο έχει υψηλότερη κεντρικότητα
    max_node = max(bc_dict, key=bc_dict.get)
    metrics['bc_max_electrode'] = ELECTRODE_NAMES[max_node]
    print(f"   Betweenness Centrality : mean={metrics['bc_mean']:.4f}, "
          f"max={metrics['bc_max']:.4f} ({metrics['bc_max_electrode']})")

    return metrics

# ============================================================
# ΒΗΜΑ 6: Υπολογισμός μετρικών για όλους τους γράφους
# ============================================================
print("\n--- ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΤΡΙΚΩΝ ---")

all_metrics = []

print("\n📊 Imagery MU:")
all_metrics.append(compute_metrics(G_imagery_mu,   'imagery',   'mu'))

print("\n📊 Imagery BETA:")
all_metrics.append(compute_metrics(G_imagery_beta, 'imagery',   'beta'))

print("\n📊 Execution MU:")
all_metrics.append(compute_metrics(G_exec_mu,      'execution', 'mu'))

print("\n📊 Execution BETA:")
all_metrics.append(compute_metrics(G_exec_beta,    'execution', 'beta'))

# ============================================================
# ΒΗΜΑ 7: Αποθήκευση σε CSV
# ============================================================
# Δημιουργούμε DataFrame από τη λίστα μετρικών
# pandas DataFrame: σαν πίνακας Excel
df = pd.DataFrame(all_metrics)

# Αποθήκευση σε CSV
csv_path = os.path.join(results_dir, "step6_metrics_S001.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV αποθηκεύτηκε: results/step6_metrics_S001.csv")

# ============================================================
# ΒΗΜΑ 8: Σύνοψη μετρικών — εκτύπωση
# ============================================================
print("\n--- ΣΥΝΟΨΗ ΜΕΤΡΙΚΩΝ ---")
print(f"\n{'Κατηγορία':<20} {'Clustering':>12} {'Efficiency':>12} {'Modularity':>12} {'BC Mean':>10}")
print("-" * 70)

for m in all_metrics:
    name = f"{m['condition']} {m['band'].upper()}"
    cc   = m['clustering_coefficient']
    ge   = m['global_efficiency']
    mod  = m['modularity'] if m['modularity'] is not None else "N/A"
    bc   = m['bc_mean']
    print(f"{name:<20} {cc:>12.4f} {ge:>12.4f} {str(mod):>12} {bc:>10.4f}")

# ============================================================
# ΒΗΜΑ 9: Γράφημα σύγκρισης μετρικών
# ============================================================
print("\nΔημιουργία γραφήματος σύγκρισης μετρικών...")

categories = ['Imagery\nMU', 'Imagery\nBETA', 'Execution\nMU', 'Execution\nBETA']
colors_img  = ['#2980B9', '#85C1E9']   # μπλε αποχρώσεις για imagery
colors_exec = ['#C0392B', '#F1948A']   # κόκκινες αποχρώσεις για execution
all_colors  = colors_img + colors_exec

cc_values = [m['clustering_coefficient'] for m in all_metrics]
ge_values = [m['global_efficiency']      for m in all_metrics]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Μετρικές Γράφων — S001\nSύγκριση Motor Imagery vs Motor Execution",
             fontsize=13, fontweight='bold')

# --- Clustering Coefficient ---
bars1 = axes[0].bar(categories, cc_values, color=all_colors,
                    edgecolor='white', linewidth=1.5, width=0.6)
axes[0].set_title("Clustering Coefficient", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Τιμή (0-1)", fontsize=10)
axes[0].set_ylim(0, max(cc_values) * 1.2)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, cc_values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

# --- Global Efficiency ---
bars2 = axes[1].bar(categories, ge_values, color=all_colors,
                    edgecolor='white', linewidth=1.5, width=0.6)
axes[1].set_title("Global Efficiency", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Τιμή (0-1)", fontsize=10)
axes[1].set_ylim(0, max(ge_values) * 1.2)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, ge_values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2980B9', label='Imagery MU'),
    Patch(facecolor='#85C1E9', label='Imagery BETA'),
    Patch(facecolor='#C0392B', label='Execution MU'),
    Patch(facecolor='#F1948A', label='Execution BETA'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()

# Αποθήκευση
metrics_plot_path = os.path.join(results_dir, "step6_metrics_plot_S001.png")
plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Γράφημα αποθηκεύτηκε: results/step6_metrics_plot_S001.png")
plt.show()

# ============================================================
# ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ
# ============================================================
print("\n" + "=" * 60)
print("🎉 ΟΛΟΚΛΗΡΩΣΗ PIPELINE ΓΙΑ S001!")
print("=" * 60)
print("\nΑρχεία που δημιουργήθηκαν στο results/:")
print("  ✅ step2_PSD_S001R03.png      — PSD γράφημα")
print("  ✅ step3_epochs_S001.png      — Epoching γράφημα")
print("  ✅ step4_connectivity_S001.png — Heatmaps")
print("  ✅ step5_graphs_S001.png      — Εγκεφαλικοί γράφοι")
print("  ✅ step6_metrics_S001.csv     — Μετρικές (CSV)")
print("  ✅ step6_metrics_plot_S001.png — Γράφημα μετρικών")
print("\nΕπόμενο βήμα:")
print("  → Quality check για όλους τους 109 subjects")
print("  → Loop για όλους τους έγκυρους subjects")
print("=" * 60)
