# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step5_graph.py
# Σκοπός  : Κατασκευή γράφων εγκεφαλικών δικτύων με NetworkX
# Βήμα    : 5 από 6
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Παίρνουμε τους δυαδικούς πίνακες 64×64 από το step4
# και τους μετατρέπουμε σε ΓΡΑΦΟΥΣ με τη βιβλιοθήκη NetworkX.
#
# Γράφος = δίκτυο από:
#   - Κόμβοι (nodes): τα 64 ηλεκτρόδια
#   - Ακμές (edges): οι συνδέσεις μεταξύ ηλεκτροδίων (r > 0.5)
#
# Φτιάχνουμε 4 γράφους:
#   G_imagery_mu, G_imagery_beta
#   G_execution_mu, G_execution_beta
#
# Οπτικοποίηση: τα ηλεκτρόδια στη θέση τους στο κεφάλι
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import numpy as np
import networkx as nx           # Για κατασκευή και ανάλυση γράφων
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ============================================================
# ΒΗΜΑ 1: Φόρτωση πινάκων από step4
# ============================================================
# Φορτώνουμε τους δυαδικούς πίνακες που αποθηκεύσαμε στο step4
# np.load(): φορτώνει αρχεία .npy

results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

print("=" * 60)
print("STEP 5 - Κατασκευή Γράφων Εγκεφαλικών Δικτύων")
print("=" * 60)

print("\nΦόρτωση πινάκων connectivity από step4...")
conn_imagery_mu_bin   = np.load(os.path.join(results_dir, "conn_imagery_mu_bin.npy"))
conn_imagery_beta_bin = np.load(os.path.join(results_dir, "conn_imagery_beta_bin.npy"))
conn_exec_mu_bin      = np.load(os.path.join(results_dir, "conn_exec_mu_bin.npy"))
conn_exec_beta_bin    = np.load(os.path.join(results_dir, "conn_exec_beta_bin.npy"))
print("✅ Πίνακες φορτώθηκαν!")
print(f"   Διαστάσεις: {conn_imagery_mu_bin.shape}")

# ============================================================
# ΒΗΜΑ 2: Ονόματα ηλεκτροδίων (64 κανάλια)
# ============================================================
# Αυτά είναι τα επίσημα ονόματα των 64 ηλεκτροδίων
# σύμφωνα με το σύστημα 10-10
# Χρησιμοποιούνται για να ονομάσουμε τους κόμβους του γράφου

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
    'O1',  'Oz',  'O2',
    'Iz'
]

# ============================================================
# ΒΗΜΑ 3: Συνάρτηση κατασκευής γράφου
# ============================================================

def build_graph(adj_matrix, electrode_names):
    """
    Μετατρέπει έναν δυαδικό πίνακα 64×64 σε γράφο NetworkX.

    Παράμετροι:
        adj_matrix     : δυαδικός πίνακας 64×64 (0 ή 1)
        electrode_names: λίστα με ονόματα ηλεκτροδίων

    Επιστρέφει:
        G: γράφος NetworkX
    """
    # nx.Graph(): δημιουργεί αμφίδρομο (undirected) γράφο
    G = nx.Graph()

    # Προσθέτουμε τους κόμβους (64 ηλεκτρόδια)
    # enumerate(): δίνει index και τιμή για κάθε στοιχείο
    for i, name in enumerate(electrode_names):
        G.add_node(i, label=name)

    # Προσθέτουμε τις ακμές (συνδέσεις)
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # i+1 για να μην διπλογράφουμε
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    return G

# ============================================================
# ΒΗΜΑ 4: Κατασκευή 4 γράφων
# ============================================================
print("\nΚατασκευή γράφων...")

G_imagery_mu   = build_graph(conn_imagery_mu_bin,   ELECTRODE_NAMES)
G_imagery_beta = build_graph(conn_imagery_beta_bin, ELECTRODE_NAMES)
G_exec_mu      = build_graph(conn_exec_mu_bin,      ELECTRODE_NAMES)
G_exec_beta    = build_graph(conn_exec_beta_bin,    ELECTRODE_NAMES)

print(f"✅ G_imagery_mu   : {G_imagery_mu.number_of_nodes()} κόμβοι, {G_imagery_mu.number_of_edges()} ακμές")
print(f"✅ G_imagery_beta : {G_imagery_beta.number_of_nodes()} κόμβοι, {G_imagery_beta.number_of_edges()} ακμές")
print(f"✅ G_exec_mu      : {G_exec_mu.number_of_nodes()} κόμβοι, {G_exec_mu.number_of_edges()} ακμές")
print(f"✅ G_exec_beta    : {G_exec_beta.number_of_nodes()} κόμβοι, {G_exec_beta.number_of_edges()} ακμές")

# ============================================================
# ΒΗΜΑ 5: Θέσεις ηλεκτροδίων στο κεφάλι
# ============================================================
# Για να οπτικοποιήσουμε τον γράφο με τα ηλεκτρόδια
# στη σωστή θέση στο κεφάλι, χρησιμοποιούμε συντεταγμένες (x, y)
# Αυτές είναι κατά προσέγγιση θέσεις για τα 64 κανάλια

def get_electrode_positions():
    """
    Επιστρέφει λεξικό {index: (x, y)} με θέσεις ηλεκτροδίων.
    Οι συντεταγμένες είναι κανονικοποιημένες (0-1).
    """
    pos = {
        # Frontal-Central
        0:  (-0.3, 0.6),   # Fc5
        1:  (-0.2, 0.65),  # Fc3
        2:  (-0.1, 0.68),  # Fc1
        3:  (0.0,  0.70),  # Fcz
        4:  (0.1,  0.68),  # Fc2
        5:  (0.2,  0.65),  # Fc4
        6:  (0.3,  0.6),   # Fc6
        # Central
        7:  (-0.4, 0.5),   # C5
        8:  (-0.25,0.5),   # C3
        9:  (-0.1, 0.5),   # C1
        10: (0.0,  0.5),   # Cz
        11: (0.1,  0.5),   # C2
        12: (0.25, 0.5),   # C4
        13: (0.4,  0.5),   # C6
        # Central-Parietal
        14: (-0.3, 0.4),   # Cp5
        15: (-0.2, 0.35),  # Cp3
        16: (-0.1, 0.32),  # Cp1
        17: (0.0,  0.30),  # Cpz
        18: (0.1,  0.32),  # Cp2
        19: (0.2,  0.35),  # Cp4
        20: (0.3,  0.4),   # Cp6
        # Frontal-Polar
        21: (-0.1, 0.95),  # Fp1
        22: (0.0,  0.97),  # Fpz
        23: (0.1,  0.95),  # Fp2
        # Anterior-Frontal
        24: (-0.35,0.85),  # Af7
        25: (-0.15,0.88),  # Af3
        26: (0.0,  0.90),  # Afz
        27: (0.15, 0.88),  # Af4
        28: (0.35, 0.85),  # Af8
        # Frontal
        29: (-0.5, 0.75),  # F7
        30: (-0.35,0.80),  # F5
        31: (-0.25,0.82),  # F3
        32: (-0.1, 0.83),  # F1
        33: (0.0,  0.84),  # Fz
        34: (0.1,  0.83),  # F2
        35: (0.25, 0.82),  # F4
        36: (0.35, 0.80),  # F6
        37: (0.5,  0.75),  # F8
        # Fronto-Temporal
        38: (-0.55,0.60),  # Ft7
        39: (0.55, 0.60),  # Ft8
        # Temporal
        40: (-0.6, 0.5),   # T7
        41: (0.6,  0.5),   # T8
        42: (-0.7, 0.5),   # T9
        43: (0.7,  0.5),   # T10
        # Temporo-Parietal
        44: (-0.55,0.4),   # Tp7
        45: (0.55, 0.4),   # Tp8
        # Parietal
        46: (-0.5, 0.25),  # P7
        47: (-0.35,0.20),  # P5
        48: (-0.25,0.18),  # P3
        49: (-0.1, 0.16),  # P1
        50: (0.0,  0.15),  # Pz
        51: (0.1,  0.16),  # P2
        52: (0.25, 0.18),  # P4
        53: (0.35, 0.20),  # P6
        54: (0.5,  0.25),  # P8
        # Parieto-Occipital
        55: (-0.3, 0.10),  # Po7
        56: (-0.15,0.08),  # Po3
        57: (0.0,  0.07),  # Poz
        58: (0.15, 0.08),  # Po4
        59: (0.3,  0.10),  # Po8
        # Occipital
        60: (-0.1, 0.03),  # O1
        61: (0.0,  0.02),  # Oz
        62: (0.1,  0.03),  # O2
        # Inion
        63: (0.0,  0.0),   # Iz
    }
    return pos

pos = get_electrode_positions()

# ============================================================
# ΒΗΜΑ 6: Οπτικοποίηση γράφων
# ============================================================
print("\nΔημιουργία γραφήματος γράφων...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Εγκεφαλικά Δίκτυα — Γράφοι S001\nΚόκκινοι κόμβοι = υψηλότερη κεντρικότητα",
             fontsize=14, fontweight='bold')

graphs_info = [
    (G_imagery_mu,   axes[0,0], "Motor IMAGERY — MU\n(8-13 Hz)",     "Blues",  "#1B6CA8"),
    (G_imagery_beta, axes[0,1], "Motor IMAGERY — BETA\n(13-30 Hz)",  "Blues",  "#1B6CA8"),
    (G_exec_mu,      axes[1,0], "Motor EXECUTION — MU\n(8-13 Hz)",   "Reds",   "#C0392B"),
    (G_exec_beta,    axes[1,1], "Motor EXECUTION — BETA\n(13-30 Hz)","Reds",   "#C0392B"),
]

for G, ax, title, cmap, edge_color in graphs_info:

    # Υπολογισμός degree centrality για χρωματισμό κόμβων
    # degree: πόσες συνδέσεις έχει κάθε κόμβος
    degree_dict = dict(G.degree())
    max_degree  = max(degree_dict.values()) if degree_dict else 1

    # Κανονικοποίηση: 0-1
    node_colors = [degree_dict[n] / max_degree for n in G.nodes()]

    # Μέγεθος κόμβων ανάλογα με τον degree
    node_sizes = [100 + 200 * (degree_dict[n] / max_degree) for n in G.nodes()]

    # Σχεδίαση γράφου
    # nx.draw_networkx(): σχεδιάζει γράφο με κόμβους και ακμές
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.15,           # Διαφάνεια ακμών (0=αόρατο, 1=πλήρες)
        edge_color=edge_color,
        width=0.5
    )

    # Σχεδίαση κόμβων με χρώμα ανάλογα με degree
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.get_cmap(cmap),
        vmin=0, vmax=1,
        alpha=0.9
    )

    # Ονόματα κόμβων (μόνο σημαντικά ηλεκτρόδια για καθαρότητα)
    important = ['C1', 'C3', 'C4', 'Cz', 'Fc3', 'Fc4', 'Cp3', 'Cp4', 'Fp1', 'Fp2', 'O1', 'O2']
    labels = {i: ELECTRODE_NAMES[i] for i in range(len(ELECTRODE_NAMES))
              if ELECTRODE_NAMES[i] in important}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight='bold')

    # Σχεδίαση κεφαλιού (κύκλος)
    circle = plt.Circle((0, 0.5), 0.52, fill=False,
                        color='gray', linewidth=2, linestyle='--')
    ax.add_patch(circle)

    # Πληροφορίες γράφου
    ax.set_title(title, fontsize=11, fontweight='bold')
    info_text = (f"Κόμβοι: {G.number_of_nodes()} | "
                 f"Ακμές: {G.number_of_edges()}")
    ax.text(0.5, -0.05, info_text, transform=ax.transAxes,
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')  # Κρύβουμε τους άξονες

plt.tight_layout()

# Αποθήκευση
graph_path = os.path.join(results_dir, "step5_graphs_S001.png")
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
print(f"✅ Γράφημα αποθηκεύτηκε: results/step5_graphs_S001.png")
plt.show()

# ============================================================
# ΒΗΜΑ 7: Εκτύπωση βασικών χαρακτηριστικών γράφων
# ============================================================
print("\n--- ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ ΓΡΑΦΩΝ ---")
print(f"{'Γράφος':<25} {'Κόμβοι':<10} {'Ακμές':<10} {'Avg Degree'}")
print("-" * 55)
for name, G in [
    ("Imagery MU",    G_imagery_mu),
    ("Imagery BETA",  G_imagery_beta),
    ("Execution MU",  G_exec_mu),
    ("Execution BETA",G_exec_beta),
]:
    avg_degree = np.mean([d for n, d in G.degree()])
    print(f"{name:<25} {G.number_of_nodes():<10} {G.number_of_edges():<10} {avg_degree:.1f}")

# ============================================================
# ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ
# ============================================================
print("\n--- ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ ---")
print("✅ 4 γράφοι εγκεφαλικών δικτύων κατασκευάστηκαν!")
print("   Έτοιμοι για υπολογισμό μετρικών στο step6.")

print("\n" + "=" * 60)
print("Επόμενο βήμα: step6_metrics.py")
print("=" * 60)
