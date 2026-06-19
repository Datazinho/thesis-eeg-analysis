# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : statistical_analysis.py
# Σκοπός  : Στατιστική σύγκριση μετρικών γράφων μεταξύ
#           Motor Imagery και Motor Execution
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Παίρνουμε τα αποτελέσματα από το loop (all_metrics_final_v2.csv)
# και κάνουμε στατιστικά τεστ για να δούμε αν οι διαφορές
# μεταξύ Imagery και Execution είναι στατιστικά σημαντικές.
#
# Χρησιμοποιούμε το Wilcoxon Signed-Rank Test:
#   - Μη παραμετρικό τεστ (δεν υποθέτει κανονική κατανομή)
#   - Κατάλληλο για paired data (ίδιοι subjects, διαφ. συνθήκες)
#   - p < 0.05 = στατιστικά σημαντική διαφορά
#
# Μετρικές που συγκρίνουμε:
#   - Clustering Coefficient
#   - Global Efficiency
#   - Modularity
#   - Betweenness Centrality (mean)
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ============================================================
# ΒΗΜΑ 1: Φόρτωση δεδομένων
# ============================================================
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

print("=" * 65)
print("STATISTICAL ANALYSIS — Imagery vs Execution")
print("=" * 65)

csv_path = os.path.join(RESULTS_DIR, "all_metrics_final_v2.csv")
df = pd.read_csv(csv_path)

print(f"\nΔεδομένα φορτώθηκαν: {len(df)} γραμμές")
print(f"Subjects: {df['subject'].nunique()}")
print(f"Κατηγορίες: {df['condition'].unique()}")
print(f"Ζώνες: {df['band'].unique()}")

# ============================================================
# ΒΗΜΑ 2: Συνάρτηση Wilcoxon test
# ============================================================
def wilcoxon_test(imagery_values, execution_values, metric_name, band):
    """
    Εκτελεί Wilcoxon Signed-Rank Test μεταξύ Imagery και Execution.

    Παράμετροι:
        imagery_values  : τιμές για Motor Imagery
        execution_values: τιμές για Motor Execution
        metric_name     : όνομα μετρικής
        band            : 'mu' ή 'beta'

    Επιστρέφει:
        dict με αποτελέσματα
    """
    # Wilcoxon test
    # alternative='two-sided': ελέγχουμε αν υπάρχει ΟΠΟΙΑΔΗΠΟΤΕ διαφορά
    statistic, p_value = stats.wilcoxon(imagery_values, execution_values,
                                         alternative='two-sided')

    # Effect size — rank-biserial correlation
    # Μετράει πόσο μεγάλη είναι η διαφορά (0=καμία, 1=μέγιστη)
    n = len(imagery_values)
    r = statistic / (n * (n + 1) / 2)
    effect_size = abs(1 - 2 * r)

    # Μέσες τιμές
    mean_imagery   = np.mean(imagery_values)
    mean_execution = np.mean(execution_values)
    diff = mean_imagery - mean_execution

    # Σημαντικότητα
    significant = p_value < 0.05

    result = {
        'metric'         : metric_name,
        'band'           : band,
        'mean_imagery'   : round(mean_imagery,   4),
        'mean_execution' : round(mean_execution, 4),
        'difference'     : round(diff,           4),
        'statistic'      : round(statistic,      4),
        'p_value'        : round(p_value,        4),
        'effect_size'    : round(effect_size,    4),
        'significant'    : significant,
        'direction'      : 'Imagery > Execution' if diff > 0 else 'Execution > Imagery'
    }

    return result

# ============================================================
# ΒΗΜΑ 3: Στατιστικά τεστ για κάθε μετρική και ζώνη
# ============================================================
metrics_to_test = [
    'clustering_coefficient',
    'global_efficiency',
    'bc_mean'
]

if 'modularity' in df.columns and df['modularity'].notna().any():
    metrics_to_test.append('modularity')

all_results = []

print(f"\n--- WILCOXON SIGNED-RANK TEST ---")
print(f"{'Μετρική':<25} {'Ζώνη':<8} {'Imagery':<10} {'Execution':<10} {'p-value':<10} {'Σημαντικό'}")
print("-" * 75)

for band in ['mu', 'beta']:
    for metric in metrics_to_test:
        # Παίρνουμε τιμές για Imagery και Execution
        # Σημαντικό: πρέπει να είναι ΙΔΙΟΙ subjects και στις δύο ομάδες
        imagery_df   = df[(df['condition']=='imagery')   & (df['band']==band)].sort_values('subject')
        execution_df = df[(df['condition']=='execution') & (df['band']==band)].sort_values('subject')

        # Κρατάμε μόνο κοινούς subjects
        common = set(imagery_df['subject']) & set(execution_df['subject'])
        imagery_df   = imagery_df[imagery_df['subject'].isin(common)]
        execution_df = execution_df[execution_df['subject'].isin(common)]

        # Κρατάμε μόνο subjects που έχουν έγκυρες τιμές ΚΑΙ στις δύο συνθήκες
        # (κρίσιμο για modularity που μπορεί να έχει NaN)
        merged = imagery_df[['subject', metric]].merge(
            execution_df[['subject', metric]],
            on='subject', suffixes=('_img', '_exc')
        ).dropna()
        img_values = merged[f'{metric}_img'].values
        exc_values = merged[f'{metric}_exc'].values

        if len(img_values) < 10:
            print(f"{metric:<25} {band:<8} Ανεπαρκή δεδομένα")
            continue

        result = wilcoxon_test(img_values, exc_values, metric, band)
        all_results.append(result)

        sig_symbol = "✅ ΝΑΙ" if result['significant'] else "❌ ΟΧΙ"
        print(f"{metric:<25} {band:<8} {result['mean_imagery']:<10} "
              f"{result['mean_execution']:<10} {result['p_value']:<10} {sig_symbol}")

# ============================================================
# ΒΗΜΑ 4: Αναλυτικά αποτελέσματα
# ============================================================
print(f"\n--- ΑΝΑΛΥΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ---")
for r in all_results:
    print(f"\n{r['metric']} — Ζώνη {r['band'].upper()}:")
    print(f"  Imagery   : {r['mean_imagery']}")
    print(f"  Execution : {r['mean_execution']}")
    print(f"  Διαφορά   : {r['difference']:+.4f} ({r['direction']})")
    print(f"  p-value   : {r['p_value']}")
    print(f"  Effect size: {r['effect_size']}")
    if r['significant']:
        print(f"  → ✅ ΣΤΑΤΙΣΤΙΚΑ ΣΗΜΑΝΤΙΚΗ ΔΙΑΦΟΡΑ (p < 0.05)")
    else:
        print(f"  → ❌ Δεν υπάρχει στατιστικά σημαντική διαφορά")

# ============================================================
# ΒΗΜΑ 5: Γραφήματα
# ============================================================
print(f"\nΔημιουργία γραφημάτων...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Στατιστική Σύγκριση — Motor Imagery vs Motor Execution\n105 Subjects",
             fontsize=14, fontweight='bold')

plot_metrics = [
    ('clustering_coefficient', 'Clustering Coefficient', axes[0,0]),
    ('global_efficiency',      'Global Efficiency',      axes[0,1]),
    ('bc_mean',                'Betweenness Centrality', axes[1,0]),
]

if 'modularity' in [r['metric'] for r in all_results]:
    plot_metrics.append(('modularity', 'Modularity', axes[1,1]))
else:
    axes[1,1].text(0.5, 0.5, 'Modularity\nN/A',
                   ha='center', va='center', transform=axes[1,1].transAxes,
                   fontsize=14, color='gray')
    axes[1,1].axis('off')

for metric, title, ax in plot_metrics:
    mu_results   = [r for r in all_results if r['metric']==metric and r['band']=='mu']
    beta_results = [r for r in all_results if r['metric']==metric and r['band']=='beta']

    if not mu_results or not beta_results:
        continue

    # Δεδομένα για violin plots
    categories = ['Imagery\nMU', 'Imagery\nBETA', 'Execution\nMU', 'Execution\nBETA']
    colors = ['#2980B9', '#85C1E9', '#C0392B', '#F1948A']

    data_to_plot = []
    for condition, band in [('imagery','mu'), ('imagery','beta'),
                             ('execution','mu'), ('execution','beta')]:
        subset = df[(df['condition']==condition) & (df['band']==band)][metric].dropna()
        data_to_plot.append(subset.values)

    # Violin plot
    parts = ax.violinplot(data_to_plot, positions=range(4),
                          showmeans=True, showmedians=True)

    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel("Τιμή", fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Προσθήκη p-value annotations
    for i, (band, x1, x2) in enumerate([('mu', 0, 2), ('beta', 1, 3)]):
        results = [r for r in all_results if r['metric']==metric and r['band']==band]
        if results:
            p = results[0]['p_value']
            sig = "*" if p < 0.05 else "ns"
            y_max = max(max(d) for d in data_to_plot if len(d) > 0)
            ax.annotate(f"p={p:.3f} {sig}",
                       xy=((x1+x2)/2, y_max * 1.02),
                       ha='center', fontsize=8,
                       color='green' if p < 0.05 else 'gray')

plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "statistical_analysis_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Γράφημα αποθηκεύτηκε: results/statistical_analysis_plots.png")
plt.show()

# ============================================================
# ΒΗΜΑ 6: Αποθήκευση αποτελεσμάτων
# ============================================================
results_df = pd.DataFrame(all_results)
results_csv = os.path.join(RESULTS_DIR, "statistical_results.csv")
results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
print(f"✅ CSV αποθηκεύτηκε: results/statistical_results.csv")

# ============================================================
# ΤΕΛΙΚΗ ΣΥΝΟΨΗ
# ============================================================
print(f"\n{'='*65}")
print("ΤΕΛΙΚΗ ΣΥΝΟΨΗ ΣΤΑΤΙΣΤΙΚΩΝ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
print(f"{'='*65}")

significant_results = [r for r in all_results if r['significant']]
non_significant     = [r for r in all_results if not r['significant']]

print(f"\nΣτατιστικά σημαντικές διαφορές (p<0.05): {len(significant_results)}")
for r in significant_results:
    print(f"  ✅ {r['metric']} ({r['band'].upper()}): p={r['p_value']}, {r['direction']}")

print(f"\nΜη σημαντικές διαφορές: {len(non_significant)}")
for r in non_significant:
    print(f"  ❌ {r['metric']} ({r['band'].upper()}): p={r['p_value']}")

print(f"\n{'='*65}")
print("Επόμενο βήμα: final_plots.py")
print(f"{'='*65}")
