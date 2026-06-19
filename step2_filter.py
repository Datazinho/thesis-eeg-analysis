# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : step2_filter.py
# Σκοπός  : Band-pass filtering του EEG σήματος
#           για τις ζώνες mu (8-13 Hz) και beta (13-30 Hz)
# Βήμα    : 2 από 6
# ============================================================
# ΤΙ ΚΑΝΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ:
# Το EEG σήμα περιέχει ΟΛΕΣ τις συχνότητες μαζί (1-100 Hz).
# Εμείς θέλουμε ΜΟΝΟ δύο συγκεκριμένες ζώνες:
#
#   Ζώνη MU   (μ): 8-13 Hz  → σχετίζεται με κινητική επεξεργασία
#   Ζώνη BETA (β): 13-30 Hz → σχετίζεται με κινητικό έλεγχο
#
# Το filtering είναι σαν να "φιλτράρουμε" τον ήχο:
# κρατάμε μόνο συγκεκριμένες νότες και πετάμε τις υπόλοιπες.
#
# Αποτέλεσμα: 2 φιλτραρισμένα σήματα (mu_raw, beta_raw)
#             + γράφημα PSD (Power Spectral Density)
# ============================================================

# --- ΒΙΒΛΙΟΘΗΚΕΣ ---
import mne
import os
import matplotlib.pyplot as plt  # Για να φτιάξουμε γραφήματα

# ============================================================
# ΒΗΜΑ 1: Ορισμός διαδρομής και φόρτωση αρχείου
# ============================================================
# Ίδια διαδρομή με το step1 — φορτώνουμε το ίδιο αρχείο

DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"

SUBJECT = "S001"
RUN     = "R03"

file_path = os.path.join(DATA_PATH, SUBJECT, f"{SUBJECT}{RUN}.edf")

print("=" * 60)
print("STEP 2 - Band-Pass Filtering")
print("=" * 60)
print(f"\nΦόρτωση αρχείου: {SUBJECT}{RUN}.edf ...")

# Φορτώνουμε το αρχείο
# preload=True: φορτώνει τα δεδομένα στη μνήμη — απαραίτητο για filtering
raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
print("✅ Αρχείο φορτώθηκε!")

# ============================================================
# ΒΗΜΑ 2: Δημιουργία ΑΝΤΙΓΡΑΦΩΝ του σήματος
# ============================================================
# Φτιάχνουμε 2 αντίγραφα του αρχικού σήματος:
#   - ένα για το mu filtering
#   - ένα για το beta filtering
#
# raw.copy() : φτιάχνει ακριβές αντίγραφο — έτσι το αρχικό
#              παραμένει αναλλοίωτο για σύγκριση

print("\nΔημιουργία αντιγράφων σήματος...")
raw_mu   = raw.copy()   # Αντίγραφο για ζώνη mu
raw_beta = raw.copy()   # Αντίγραφο για ζώνη beta
print("✅ Αντίγραφα δημιουργήθηκαν!")

# ============================================================
# ΒΗΜΑ 3: Band-Pass Filtering για ζώνη MU (8-13 Hz)
# ============================================================
# raw.filter(l_freq, h_freq):
#   l_freq = lower frequency  = κατώτατη συχνότητα (8 Hz)
#   h_freq = higher frequency = ανώτατη συχνότητα (13 Hz)
#
# Αποτέλεσμα: κρατάμε ΜΟΝΟ συχνότητες μεταξύ 8 και 13 Hz
# method='fir': τύπος φίλτρου — FIR (Finite Impulse Response)
#               είναι ο πιο συνηθισμένος για EEG
# verbose=False: δεν εκτυπώνει πολλά μηνύματα

print("\nΦιλτράρισμα ζώνης MU (8-13 Hz)...")
raw_mu.filter(l_freq=8.0, h_freq=13.0, method='fir', verbose=False)
print("✅ Ζώνη MU φιλτραρίστηκε!")

# ============================================================
# ΒΗΜΑ 4: Band-Pass Filtering για ζώνη BETA (13-30 Hz)
# ============================================================
# Ίδια διαδικασία αλλά για συχνότητες 13-30 Hz

print("\nΦιλτράρισμα ζώνης BETA (13-30 Hz)...")
raw_beta.filter(l_freq=13.0, h_freq=30.0, method='fir', verbose=False)
print("✅ Ζώνη BETA φιλτραρίστηκε!")

# ============================================================
# ΒΗΜΑ 5: Εκτύπωση αποτελεσμάτων
# ============================================================
# Ελέγχουμε ότι τα φιλτραρισμένα σήματα έχουν τα σωστά χαρακτηριστικά

print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ FILTERING ---")
print(f"Αρχικό σήμα    : {raw.info['sfreq']} Hz, {len(raw.ch_names)} κανάλια")
print(f"Ζώνη MU (φ.)   : {raw_mu.info['sfreq']} Hz, {len(raw_mu.ch_names)} κανάλια")
print(f"Ζώνη BETA (φ.) : {raw_beta.info['sfreq']} Hz, {len(raw_beta.ch_names)} κανάλια")

# ============================================================
# ΒΗΜΑ 6: Δημιουργία γραφήματος PSD
# ============================================================
# PSD = Power Spectral Density (Φασματική Πυκνότητα Ισχύος)
# Δείχνει πόση "ενέργεια" έχει το σήμα σε κάθε συχνότητα.
#
# Θα φτιάξουμε 3 γραφήματα δίπλα-δίπλα:
#   1. Αρχικό σήμα (όλες οι συχνότητες)
#   2. Φιλτραρισμένο MU (8-13 Hz)
#   3. Φιλτραρισμένο BETA (13-30 Hz)

print("\nΔημιουργία γραφήματος PSD...")

# fig, axes: δημιουργεί παράθυρο με 3 γραφήματα δίπλα-δίπλα
# figsize=(15,4): πλάτος 15, ύψος 4 ίντσες
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# --- Γράφημα 1: Αρχικό σήμα ---
# compute_psd(): υπολογίζει το PSD
# fmax=50: δείχνει συχνότητες μέχρι 50 Hz
# picks='eeg': μόνο τα EEG κανάλια (όχι άλλα)
psd_raw = raw.compute_psd(fmax=50, picks='eeg', verbose=False)
psd_raw.plot(axes=axes[0], show=False)
axes[0].set_title("Αρχικό σήμα\n(όλες οι συχνότητες)", fontsize=11)
# Προσθέτουμε κάθετες γραμμές για να δείξουμε τις ζώνες
axes[0].axvspan(8, 13, alpha=0.2, color='blue', label='MU')
axes[0].axvspan(13, 30, alpha=0.2, color='red', label='BETA')
axes[0].legend(fontsize=9)

# --- Γράφημα 2: Φιλτραρισμένο MU ---
psd_mu = raw_mu.compute_psd(fmax=50, picks='eeg', verbose=False)
psd_mu.plot(axes=axes[1], show=False)
axes[1].set_title("Φιλτραρισμένο MU\n(8-13 Hz)", fontsize=11)
axes[1].axvspan(8, 13, alpha=0.2, color='blue', label='MU')
axes[1].legend(fontsize=9)

# --- Γράφημα 3: Φιλτραρισμένο BETA ---
psd_beta = raw_beta.compute_psd(fmax=50, picks='eeg', verbose=False)
psd_beta.plot(axes=axes[2], show=False)
axes[2].set_title("Φιλτραρισμένο BETA\n(13-30 Hz)", fontsize=11)
axes[2].axvspan(13, 30, alpha=0.2, color='red', label='BETA')
axes[2].legend(fontsize=9)

# Τίτλος για όλο το γράφημα
fig.suptitle("Power Spectral Density - S001R03\nΠριν και Μετά το Filtering",
             fontsize=13, fontweight='bold')

# Καλύτερη εμφάνιση — αποφεύγει επικάλυψη
plt.tight_layout()

# ============================================================
# ΒΗΜΑ 7: Αποθήκευση γραφήματος
# ============================================================
# Αποθηκεύουμε το γράφημα στον φάκελο results
# ώστε να το στείλουμε στον καθηγητή

# Βρίσκουμε τον φάκελο results σχετικά με το τρέχον directory
results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "results", "step2_PSD_S001R03.png")

# Αποθηκεύουμε
# dpi=150: ανάλυση εικόνας (150 dots per inch — καλή ποιότητα)
# bbox_inches='tight': κόβει τα κενά γύρω από το γράφημα
plt.savefig(results_path, dpi=150, bbox_inches='tight')
print(f"✅ Γράφημα αποθηκεύτηκε: results/step2_PSD_S001R03.png")

# Εμφανίζουμε το γράφημα στην οθόνη
plt.show()

print("\n--- ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ ---")
print("✅ Filtering ολοκληρώθηκε!")
print("   - raw_mu   : φιλτραρισμένο σήμα ζώνης MU (8-13 Hz)")
print("   - raw_beta : φιλτραρισμένο σήμα ζώνης BETA (13-30 Hz)")
print("   - Γράφημα PSD αποθηκεύτηκε στον φάκελο results/")

print("\n" + "=" * 60)
print("Επόμενο βήμα: step3_epochs.py")
print("=" * 60)
