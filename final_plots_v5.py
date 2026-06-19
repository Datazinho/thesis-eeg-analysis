# ============================================================
# ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ - Ανάλυση EEG με Θεωρία Γράφων
# ============================================================
# Αρχείο  : final_plots_v5.py
# Έκδοση  : v5 — ΤΕΛΙΚΗ ΕΚΔΟΣΗ
# Αρχεία  : FINAL_CH4_Fig1_PSD.png κλπ
# ============================================================
# ΑΛΛΑΓΕΣ ΑΠΟ v4:
#   ✅ Ονόματα: FINAL_CH4/CH5_FigX (ξεκάθαρα τελικά)
#   ✅ PSD: lw=1.5 + Beta Band μέχρι 50Hz
#   ✅ Epochs: lw=1.5
#   ✅ Brain graphs: ακμές alpha=0.30
#   ✅ CH5_Fig1: BC y-axis fix (ylim)
#   ✅ CH5_Fig5: legend εκτός γραφήματος (bbox_to_anchor)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy import stats
import mne
import os

# ════════════════════════════════════════════════════════════
# ΠΑΛΕΤΑ ΧΡΩΜΑΤΩΝ — IBM Colorblind Safe (ΤΕΛΙΚΗ)
# ════════════════════════════════════════════════════════════
C = {
    'imagery'     : '#0072B2',
    'execution'   : '#E69F00',
    'imagery_mu'  : '#0072B2',
    'imagery_beta': '#56B4E9',
    'exec_mu'     : '#E69F00',
    'exec_beta'   : '#D55E00',
    'accent'      : '#CC79A7',
    'original'    : '#555555',
    'text'        : '#2C2C2C',
}

def set_style():
    plt.rcParams.update({
        'font.family'       : 'DejaVu Sans',
        'font.size'         : 11,
        'axes.titlesize'    : 13,
        'axes.labelsize'    : 11,
        'axes.titleweight'  : 'bold',
        'axes.facecolor'    : '#FFFFFF',
        'axes.grid'         : False,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'figure.facecolor'  : 'white',
        'xtick.labelsize'   : 10,
        'ytick.labelsize'   : 10,
        'legend.fontsize'   : 10,
        'legend.framealpha' : 0.9,
        'legend.edgecolor'  : '#CCCCCC',
    })

set_style()

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
DATA_PATH = r"C:\Users\ctsio\OneDrive\Desktop\BNP-My Thesis\me matzako\dataset"
df = pd.read_csv(os.path.join(RESULTS_DIR, "all_metrics_final_v2.csv"))

def save(name):
    """Αποθηκεύει με FINAL_ prefix."""
    path = os.path.join(RESULTS_DIR, f"FINAL_{name}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✅ FINAL_{name}.png")
    plt.show()

print("=" * 65)
print("FINAL PLOTS v5 — ΤΕΛΙΚΗ ΕΚΔΟΣΗ")
print("=" * 65)

# ════════════════════════════════════════════════════════════
# FINAL_CH4_Fig1 — PSD
# ════════════════════════════════════════════════════════════
print("\n[1/10] FINAL_CH4_Fig1 — PSD...")

fp  = os.path.join(DATA_PATH, "S001", "S001R03.edf")
raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
raw_mu   = raw.copy().filter(8.0,  13.0, method='fir', verbose=False)
raw_beta = raw.copy().filter(13.0, 30.0, method='fir', verbose=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('white')
fig.suptitle("Power Spectral Density — S001 R03\nBefore and After Band-Pass Filtering",
             fontsize=14, fontweight='bold', color=C['text'])

for ax, raw_obj, title, lc, span in [
    (axes[0], raw,      "Original Signal\n(All Frequencies)", C['original'], None),
    (axes[1], raw_mu,   "Filtered — Mu Band\n(8–13 Hz)",      C['imagery'],  (8,  13)),
    (axes[2], raw_beta, "Filtered — Beta Band\n(13–30 Hz)",   C['exec_beta'],(13, 30)),
]:
    psd_obj  = raw_obj.compute_psd(fmax=50, picks='eeg', verbose=False)
    freqs    = psd_obj.freqs
    psd_data = psd_obj.get_data()
    psd_mean = 10 * np.log10(np.mean(psd_data, axis=0))
    psd_sem  = stats.sem(10 * np.log10(psd_data + 1e-30), axis=0)

    ax.plot(freqs, psd_mean, color=lc, lw=1.5)          # ← lw=1.5
    ax.fill_between(freqs, psd_mean-psd_sem, psd_mean+psd_sem,
                    color=lc, alpha=0.18)
    if span:
        ax.axvspan(span[0], span[1], alpha=0.15, color=lc)
        ax.axvline(span[0], color=lc, lw=1, ls='--', alpha=0.6)
        ax.axvline(span[1], color=lc, lw=1, ls='--', alpha=0.6)
    ax.set_xlim(0, 50)                                   # ← μέχρι 50Hz παντού
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Power (dB)",     fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
save("CH4_Fig1_PSD")

# ════════════════════════════════════════════════════════════
# FINAL_CH4_Fig2 — Epochs
# ════════════════════════════════════════════════════════════
print("[2/10] FINAL_CH4_Fig2 — Epochs...")

def get_ep_single(subject, run, band):
    fp  = os.path.join(DATA_PATH, subject, f"{subject}{run}.edf")
    if not os.path.exists(fp): return None
    try:
        raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
        raw.filter(8.0 if band=='mu' else 13.0,
                   13.0 if band=='mu' else 30.0,
                   method='fir', verbose=False)
        events, eid = mne.events_from_annotations(raw, verbose=False)
        wanted = {k: v for k, v in eid.items() if 'T1' in k or 'T2' in k}
        if not wanted: return None
        ep = mne.Epochs(raw, events, event_id=wanted, tmin=-1.0, tmax=3.0,
                        baseline=(None,0), preload=True, verbose=False)
        return ep if len(ep) > 0 else None
    except: return None

def get_ep_all(subject, runs, band):
    all_ep = [e for r in runs for e in [get_ep_single(subject, r, band)] if e is not None]
    if not all_ep: return None
    return mne.concatenate_epochs(all_ep, verbose=False) if len(all_ep)>1 else all_ep[0]

RUNS_IMG = ["R03","R04","R07","R08","R11","R12"]
RUNS_EXC = ["R05","R06","R09","R10","R13","R14"]

ep_img = get_ep_all("S001", RUNS_IMG, "mu")
ep_exc = get_ep_all("S001", RUNS_EXC, "mu")
chs    = ep_img.ch_names
ch_idx = chs.index('C3.') if 'C3.' in chs else 0
times  = ep_img.times
mi = np.mean(ep_img.get_data()[:,ch_idx,:],axis=0)*1e6
me = np.mean(ep_exc.get_data()[:,ch_idx,:],axis=0)*1e6
si = stats.sem(ep_img.get_data()[:,ch_idx,:],axis=0)*1e6
se = stats.sem(ep_exc.get_data()[:,ch_idx,:],axis=0)*1e6

fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.patch.set_facecolor('white')
fig.suptitle("EEG Epoch Comparison — S001 | Channel C3 | Mu Band (8–13 Hz)",
             fontsize=14,fontweight='bold',color=C['text'])

ax = axes[0]
ax.plot(times, mi, color=C['imagery'],   lw=1.5, label='Motor Imagery')   # ← lw=1.5
ax.plot(times, me, color=C['execution'], lw=1.5, label='Motor Execution') # ← lw=1.5
ax.fill_between(times, mi-si, mi+si, color=C['imagery'],   alpha=0.18)
ax.fill_between(times, me-se, me+se, color=C['execution'], alpha=0.18)
ax.axvline(x=0, color=C['text'], lw=1.2, ls='--', alpha=0.7,
           label='Movement onset (t=0)')
ax.axvspan(-1,0,alpha=0.05,color='gray',label='Baseline')
ax.set_xlabel("Time (s)",fontsize=11)
ax.set_ylabel("Amplitude (μV)",fontsize=11)
ax.set_title("Mean EEG Signal ± SEM",fontsize=12,fontweight='bold')
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = axes[1]
counts = [len(ep_img),len(ep_exc)]
bars = ax2.bar(['Motor Imagery','Motor Execution'],counts,
               color=[C['imagery'],C['execution']],
               width=0.5,edgecolor='white',linewidth=1.5,alpha=0.88)
for bar,count in zip(bars,counts):
    ax2.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.3,str(count),
             ha='center',fontsize=13,fontweight='bold')
ax2.set_ylabel("Number of Epochs",fontsize=11)
ax2.set_title("Epochs per Condition",fontsize=12,fontweight='bold')
ax2.set_ylim(0,max(counts)*1.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
save("CH4_Fig2_epochs")

# ════════════════════════════════════════════════════════════
# FINAL_CH4_Fig3 — Connectivity
# ════════════════════════════════════════════════════════════
print("[3/10] FINAL_CH4_Fig3 — Connectivity...")

def get_conn_all(subject, runs, band):
    all_epochs = [e for r in runs for e in [get_ep_single(subject, r, band)] if e is not None]
    if not all_epochs: return None
    epochs = mne.concatenate_epochs(all_epochs, verbose=False) if len(all_epochs)>1 else all_epochs[0]
    data   = epochs.get_data()
    all_c  = np.zeros((data.shape[0],64,64))
    for i in range(data.shape[0]):
        all_c[i] = np.abs(np.corrcoef(data[i]))
    mean_c = np.mean(all_c,axis=0)
    np.fill_diagonal(mean_c,0)
    return mean_c

print("   Υπολογισμός connectivity (όλα τα runs)...")
c_img_mu   = get_conn_all("S001", RUNS_IMG, "mu")
c_img_beta = get_conn_all("S001", RUNS_IMG, "beta")
c_exc_mu   = get_conn_all("S001", RUNS_EXC, "mu")
c_exc_beta = get_conn_all("S001", RUNS_EXC, "beta")

fig,axes = plt.subplots(2,2,figsize=(14,12))
fig.patch.set_facecolor('white')
fig.suptitle("Functional Connectivity Matrices 64×64 — S001\nPearson Correlation (Continuous Values)",
             fontsize=14,fontweight='bold',color=C['text'])

for matrix,title,cmap,ax in [
    (c_img_mu,  "Motor Imagery — Mu Band\n(8–13 Hz)",    'Blues',  axes[0,0]),
    (c_img_beta,"Motor Imagery — Beta Band\n(13–30 Hz)", 'Blues',  axes[0,1]),
    (c_exc_mu,  "Motor Execution — Mu Band\n(8–13 Hz)",  'Oranges',axes[1,0]),
    (c_exc_beta,"Motor Execution — Beta Band\n(13–30 Hz)",'Oranges',axes[1,1]),
]:
    im = ax.imshow(matrix,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label='Pearson r')
    ax.set_title(title,fontsize=11,fontweight='bold',pad=8)
    ax.set_xlabel("Electrode index",fontsize=9)
    ax.set_ylabel("Electrode index",fontsize=9)
    mr = np.mean(matrix[matrix>0])
    ax.text(0.02,0.98,f"Mean r = {mr:.3f}",
            transform=ax.transAxes,fontsize=9,va='top',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.85,edgecolor='#CCCCCC'))

plt.tight_layout(pad=2.5)
save("CH4_Fig3_connectivity")

# ════════════════════════════════════════════════════════════
# FINAL_CH4_Fig4 — Brain Graphs (ακμές alpha=0.30)
# ════════════════════════════════════════════════════════════
print("[4/10] FINAL_CH4_Fig4 — Brain graphs...")

EN = [
    'Fc5','Fc3','Fc1','Fcz','Fc2','Fc4','Fc6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'Cp5','Cp3','Cp1','Cpz','Cp2','Cp4','Cp6',
    'Fp1','Fpz','Fp2','Af7','Af3','Afz','Af4','Af8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'Ft7','Ft8','T7','T8','T9','T10','Tp7','Tp8',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'Po7','Po3','Poz','Po4','Po8','O1','Oz','O2','Iz'
]

pos = {
    0:(-0.3,0.6),1:(-0.2,0.65),2:(-0.1,0.68),3:(0.0,0.70),4:(0.1,0.68),5:(0.2,0.65),6:(0.3,0.6),
    7:(-0.4,0.5),8:(-0.25,0.5),9:(-0.1,0.5),10:(0.0,0.5),11:(0.1,0.5),12:(0.25,0.5),13:(0.4,0.5),
    14:(-0.3,0.4),15:(-0.2,0.35),16:(-0.1,0.32),17:(0.0,0.30),18:(0.1,0.32),19:(0.2,0.35),20:(0.3,0.4),
    21:(-0.1,0.95),22:(0.0,0.97),23:(0.1,0.95),
    24:(-0.35,0.85),25:(-0.15,0.88),26:(0.0,0.90),27:(0.15,0.88),28:(0.35,0.85),
    29:(-0.5,0.75),30:(-0.35,0.80),31:(-0.25,0.82),32:(-0.1,0.83),33:(0.0,0.84),
    34:(0.1,0.83),35:(0.25,0.82),36:(0.35,0.80),37:(0.5,0.75),
    38:(-0.55,0.60),39:(0.55,0.60),40:(-0.6,0.5),41:(0.6,0.5),42:(-0.7,0.5),43:(0.7,0.5),
    44:(-0.55,0.4),45:(0.55,0.4),
    46:(-0.5,0.25),47:(-0.35,0.20),48:(-0.25,0.18),49:(-0.1,0.16),50:(0.0,0.15),
    51:(0.1,0.16),52:(0.25,0.18),53:(0.35,0.20),54:(0.5,0.25),
    55:(-0.3,0.10),56:(-0.15,0.08),57:(0.0,0.07),58:(0.15,0.08),59:(0.3,0.10),
    60:(-0.1,0.03),61:(0.0,0.02),62:(0.1,0.03),63:(0.0,0.0),
}

def build_G(m,thr=0.5):
    G=nx.Graph()
    for i in range(64): G.add_node(i)
    for i in range(64):
        for j in range(i+1,64):
            if m[i,j]>thr: G.add_edge(i,j)
    return G

G_im=build_G(c_img_mu); G_ib=build_G(c_img_beta)
G_em=build_G(c_exc_mu); G_eb=build_G(c_exc_beta)

fig,axes=plt.subplots(2,2,figsize=(16,14))
fig.patch.set_facecolor('white')
fig.suptitle("Brain Functional Networks — S001\nNode color = Betweenness Centrality",
             fontsize=14,fontweight='bold',color=C['text'])

for G,ax,title,cmap,ec in [
    (G_im,axes[0,0],"Motor Imagery — Mu Band\n(8–13 Hz)",  'Blues',  C['imagery']),
    (G_ib,axes[0,1],"Motor Imagery — Beta Band\n(13–30 Hz)",'Blues', C['imagery']),
    (G_em,axes[1,0],"Motor Execution — Mu Band\n(8–13 Hz)",'Oranges',C['execution']),
    (G_eb,axes[1,1],"Motor Execution — Beta Band\n(13–30 Hz)",'Oranges',C['exec_beta']),
]:
    bc  = nx.betweenness_centrality(G,normalized=True)
    deg = dict(G.degree())
    m   = max(bc.values()) if bc else 1
    nc  = [bc.get(n,0)/m for n in G.nodes()]
    ns  = [80+180*(deg.get(n,0)/max(deg.values())) for n in G.nodes()] if deg else [100]*64

    nx.draw_networkx_edges(G,pos,ax=ax,alpha=0.30,       # ← alpha=0.30
                           edge_color=ec,width=0.5)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_size=ns,
                           node_color=nc,cmap=plt.cm.get_cmap(cmap),
                           vmin=0,vmax=1,alpha=0.9)
    imp=['C1','C3','C4','Cz','Fc3','Fc4','Cp3','Cp4','Fp1','Fp2','O1','O2']
    lbl={i:EN[i] for i in range(64) if EN[i] in imp}
    nx.draw_networkx_labels(G,pos,lbl,ax=ax,font_size=7,
                            font_weight='bold',font_color=C['text'])
    ax.add_patch(plt.Circle((0,0.5),0.52,fill=False,
                            color='#AAAAAA',linewidth=1.5,linestyle='--'))
    ax.set_title(title,fontsize=11,fontweight='bold',pad=8)
    ax.text(0.5,-0.04,f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}",
            transform=ax.transAxes,ha='center',fontsize=9,
            bbox=dict(boxstyle='round',facecolor='white',edgecolor='#CCCCCC',alpha=0.9))
    ax.set_xlim(-0.85,0.85); ax.set_ylim(-0.1,1.1); ax.axis('off')

plt.tight_layout(pad=2.0)
save("CH4_Fig4_brain_graphs")

# ════════════════════════════════════════════════════════════
# FINAL_CH4_Fig5 — Metrics S001
# ════════════════════════════════════════════════════════════
print("[5/10] FINAL_CH4_Fig5 — Metrics S001...")

s001={'Imagery MU':{'CC':0.8776,'GE':0.8911,'BC':0.0035},
      'Imagery BETA':{'CC':0.8084,'GE':0.7863,'BC':0.0060},
      'Execution MU':{'CC':0.8748,'GE':0.8854,'BC':0.0037},
      'Execution BETA':{'CC':0.8126,'GE':0.7872,'BC':0.0060}}
cats=[*s001]; cols=[C['imagery_mu'],C['imagery_beta'],C['exec_mu'],C['exec_beta']]

fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.patch.set_facecolor('white')
fig.suptitle("Graph Metrics — S001\nMotor Imagery vs Motor Execution",
             fontsize=14,fontweight='bold',color=C['text'])

for ax,metric,ylabel in [
    (axes[0],'CC','Clustering Coefficient'),
    (axes[1],'GE','Global Efficiency'),
    (axes[2],'BC','Betweenness Centrality'),
]:
    vals=[s001[c][metric] for c in cats]
    bars=ax.bar(cats,vals,color=cols,edgecolor='white',
                linewidth=1.5,width=0.55,alpha=0.88)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(vals)*0.01,
                f'{val:.4f}' if metric=='BC' else f'{val:.3f}',
                ha='center',fontsize=10,fontweight='bold')
    ax.set_ylabel(ylabel,fontsize=11)
    ax.set_title(ylabel+" — S001",fontsize=12,fontweight='bold')
    ax.set_ylim(0,max(vals)*1.18)
    ax.tick_params(axis='x',rotation=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

leg=[mpatches.Patch(color=c,label=l) for c,l in zip(cols,cats)]
fig.legend(handles=leg,loc='lower center',ncol=4,
           fontsize=10,bbox_to_anchor=(0.5,-0.05))
plt.tight_layout(pad=2.0)
save("CH4_Fig5_metrics_S001")

# ════════════════════════════════════════════════════════════
# FINAL_CH5_Fig1 — Metrics Comparison (BC y-axis fix)
# ════════════════════════════════════════════════════════════
print("[6/10] FINAL_CH5_Fig1 — Metrics comparison...")

metrics_info={'clustering_coefficient':'Clustering\nCoefficient',
              'global_efficiency':'Global\nEfficiency',
              'modularity':'Modularity'}

fig,axes=plt.subplots(1,3,figsize=(18,7))
fig.patch.set_facecolor('white')
fig.suptitle("Graph Metrics: Motor Imagery vs Motor Execution\n105 Subjects | Mean ± SEM",
             fontsize=14,fontweight='bold',color=C['text'])

for ax_i,(band,blabel) in enumerate([('mu','Mu Band (8–13 Hz)'),
                                       ('beta','Beta Band (13–30 Hz)')]):
    ax=axes[ax_i]; x=np.arange(len(metrics_info)); w=0.35
    im,is_,em,es_=[],[],[],[]
    for m in metrics_info:
        img=df[(df['condition']=='imagery')  &(df['band']==band)][m].dropna()
        exc=df[(df['condition']=='execution')&(df['band']==band)][m].dropna()
        im.append(img.mean()); is_.append(img.sem())
        em.append(exc.mean()); es_.append(exc.sem())

    ax.bar(x-w/2,im,w,yerr=is_,capsize=5,color=C['imagery'],alpha=0.88,
           label='Motor Imagery',edgecolor='white',linewidth=1.2,
           error_kw={'elinewidth':1.5,'ecolor':'#004A8F'})
    ax.bar(x+w/2,em,w,yerr=es_,capsize=5,color=C['execution'],alpha=0.88,
           label='Motor Execution',edgecolor='white',linewidth=1.2,
           error_kw={'elinewidth':1.5,'ecolor':'#A06B00'})
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_info.values()),fontsize=10)
    ax.set_ylabel("Value",fontsize=11)
    ax.set_title(blabel,fontsize=13,fontweight='bold',pad=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if band=='mu':
        yt=max(im[0],em[0])+is_[0]+0.015
        ax.annotate('★  p = 0.041',xy=(x[0],yt),ha='center',
                    fontsize=12,color=C['accent'],fontweight='bold')

# BC panel — ξεχωριστό y-axis με σωστό ylim
ax3=axes[2]
all_bc_vals=[]
for bi,(band,bcol_i,bcol_e) in enumerate([
    ('mu',  C['imagery_mu'],  C['exec_mu']),
    ('beta', C['imagery_beta'],C['exec_beta']),
]):
    img_bc=df[(df['condition']=='imagery')  &(df['band']==band)]['bc_mean'].dropna()
    exc_bc=df[(df['condition']=='execution')&(df['band']==band)]['bc_mean'].dropna()
    all_bc_vals.extend([img_bc.mean()+img_bc.sem(), exc_bc.mean()+exc_bc.sem()])
    off=bi*2.5
    ax3.bar(off,   img_bc.mean(),0.8,yerr=img_bc.sem(),capsize=5,
            color=bcol_i,alpha=0.88,edgecolor='white',
            error_kw={'elinewidth':1.5})
    ax3.bar(off+0.9,exc_bc.mean(),0.8,yerr=exc_bc.sem(),capsize=5,
            color=bcol_e,alpha=0.88,edgecolor='white',
            error_kw={'elinewidth':1.5})
    ax3.text(off+0.45,-0.0007,'Mu' if band=='mu' else 'Beta',
             ha='center',fontsize=10,fontweight='bold',color=C['text'])

ax3.set_ylim(0, max(all_bc_vals)*1.25)   # ← fix ylim
ax3.set_xticks([])
ax3.set_ylabel("Betweenness Centrality",fontsize=11)
ax3.set_title("Betweenness Centrality\nMu vs Beta Band",fontsize=12,fontweight='bold')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
leg3=[mpatches.Patch(color=C['imagery_mu'],  label='Imagery MU'),
      mpatches.Patch(color=C['imagery_beta'], label='Imagery BETA'),
      mpatches.Patch(color=C['exec_mu'],      label='Execution MU'),
      mpatches.Patch(color=C['exec_beta'],    label='Execution BETA')]
ax3.legend(handles=leg3,fontsize=9,loc='upper right')

plt.tight_layout(pad=2.0)
save("CH5_Fig1_metrics_comparison")

# ════════════════════════════════════════════════════════════
# FINAL_CH5_Fig2 — CC MU Boxplot (κύριο εύρημα)
# ════════════════════════════════════════════════════════════
print("[7/10] FINAL_CH5_Fig2 — CC MU boxplot...")

img_cc=df[(df['condition']=='imagery')  &(df['band']=='mu')]['clustering_coefficient'].dropna()
exc_cc=df[(df['condition']=='execution')&(df['band']=='mu')]['clustering_coefficient'].dropna()

fig,ax=plt.subplots(figsize=(9,7))
fig.patch.set_facecolor('white')
bp=ax.boxplot([img_cc.values,exc_cc.values],patch_artist=True,widths=0.45,
              medianprops=dict(color='white',linewidth=2.5),
              whiskerprops=dict(linewidth=1.5,color='#555555'),
              capprops=dict(linewidth=1.5,color='#555555'),
              flierprops=dict(marker='o',markersize=4,alpha=0.3,markerfacecolor='gray'))
bp['boxes'][0].set_facecolor(C['imagery']);   bp['boxes'][0].set_alpha(0.85)
bp['boxes'][1].set_facecolor(C['execution']); bp['boxes'][1].set_alpha(0.85)

np.random.seed(42)
for i,(data,color) in enumerate([(img_cc,'#004A8F'),(exc_cc,'#A06B00')]):
    jitter=np.random.normal(i+1,0.06,len(data))
    ax.scatter(jitter,data,alpha=0.22,color=color,s=16,zorder=2)

ax.set_xticks([1,2])
ax.set_xticklabels(['Motor Imagery','Motor Execution'],fontsize=12)
ax.set_ylabel("Clustering Coefficient",fontsize=12)
ax.set_title("Clustering Coefficient — MU Band (8–13 Hz)\n"
             "Statistically Significant Difference  (Wilcoxon, p = 0.041)",
             fontsize=13,fontweight='bold',pad=12)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

y_max=max(img_cc.max(),exc_cc.max())+0.02
ax.plot([1,2],[y_max,y_max],color=C['text'],lw=1.5)
ax.text(1.5,y_max+0.005,'★  p = 0.041',
        ha='center',fontsize=13,color=C['accent'],fontweight='bold')
ax.text(0.03,0.97,
        f"Imagery:    M = {img_cc.mean():.4f} ± {img_cc.std():.4f}\n"
        f"Execution: M = {exc_cc.mean():.4f} ± {exc_cc.std():.4f}\n"
        f"Wilcoxon:  p = 0.0410\nEffect size (r) = 0.2300",
        transform=ax.transAxes,fontsize=10,va='top',
        bbox=dict(boxstyle='round,pad=0.5',facecolor='white',
                  edgecolor='#CCCCCC',alpha=0.95))

plt.tight_layout(pad=2.0)
save("CH5_Fig2_cc_mu_boxplot")

# ════════════════════════════════════════════════════════════
# FINAL_CH5_Fig3 — MU vs BETA
# ════════════════════════════════════════════════════════════
print("[8/10] FINAL_CH5_Fig3 — MU vs BETA...")

mt={'clustering_coefficient':'Clustering Coefficient',
    'global_efficiency':'Global Efficiency',
    'bc_mean':'Betweenness Centrality',
    'modularity':'Modularity'}
gcols=[C['imagery_mu'],C['imagery_beta'],C['exec_mu'],C['exec_beta']]
glabs=['Imagery\nMU','Imagery\nBETA','Execution\nMU','Execution\nBETA']

fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.patch.set_facecolor('white')
fig.suptitle("Mu Band vs Beta Band — All Metrics\n105 Subjects",
             fontsize=14,fontweight='bold',color=C['text'])

for idx,(metric,title) in enumerate(mt.items()):
    ax=axes.flatten()[idx]
    groups=[]
    for cond,band in [('imagery','mu'),('imagery','beta'),
                       ('execution','mu'),('execution','beta')]:
        groups.append(df[(df['condition']==cond)&(df['band']==band)][metric].dropna().values)
    bp=ax.boxplot(groups,patch_artist=True,widths=0.45,
                  medianprops=dict(color='white',linewidth=2),
                  whiskerprops=dict(linewidth=1.2,color='#555555'),
                  capprops=dict(linewidth=1.2,color='#555555'))
    for patch,color in zip(bp['boxes'],gcols):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax.set_xticklabels(glabs,fontsize=9)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=8)
    ax.set_ylabel("Value",fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if metric=='clustering_coefficient':
        ax.text(0.5,0.02,'★  MU: p = 0.041 (significant)',
                transform=ax.transAxes,ha='center',
                fontsize=9,color=C['accent'],fontweight='bold')

plt.tight_layout(pad=2.5)
save("CH5_Fig3_mu_vs_beta")

# ════════════════════════════════════════════════════════════
# FINAL_CH5_Fig4 — Correlation Heatmap
# ════════════════════════════════════════════════════════════
print("[9/10] FINAL_CH5_Fig4 — Heatmap...")

mc=['clustering_coefficient','global_efficiency','bc_mean','modularity']
ml=['CC','GE','BC','Mod']

fig,axes=plt.subplots(1,2,figsize=(14,6))
fig.patch.set_facecolor('white')
fig.suptitle("Metric Correlation Heatmap\nPearson Correlation Between Graph Metrics | 105 Subjects",
             fontsize=14,fontweight='bold',color=C['text'])

for ax_i,(cond,title,cmap) in enumerate([
    ('imagery','Motor Imagery','Blues'),
    ('execution','Motor Execution','Oranges')
]):
    ax=axes[ax_i]
    sub=df[df['condition']==cond][mc].dropna()
    corr=sub.corr()
    im=ax.imshow(corr.values,cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label='Pearson r')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(ml,fontsize=11); ax.set_yticklabels(ml,fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=10)
    for i in range(4):
        for j in range(4):
            v=corr.values[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',
                    fontsize=11,fontweight='bold',
                    color='white' if abs(v)>0.6 else C['text'])

plt.tight_layout(pad=2.0)
save("CH5_Fig4_correlation_heatmap")

# ════════════════════════════════════════════════════════════
# FINAL_CH5_Fig5 — Statistical Summary (legend fix)
# ════════════════════════════════════════════════════════════
print("[10/10] FINAL_CH5_Fig5 — Statistical summary...")

stat={
    'clustering_coefficient':{'mu':{'img':0.835,'exc':0.838,'p':0.041,'sig':True},
                              'beta':{'img':0.782,'exc':0.783,'p':0.975,'sig':False}},
    'global_efficiency':     {'mu':{'img':0.820,'exc':0.822,'p':0.660,'sig':False},
                              'beta':{'img':0.733,'exc':0.734,'p':0.805,'sig':False}},
    'bc_mean':               {'mu':{'img':0.0056,'exc':0.0058,'p':0.475,'sig':False},
                              'beta':{'img':0.0086,'exc':0.0083,'p':0.239,'sig':False}},
    'modularity':            {'mu':{'img':0.143, 'exc':0.146, 'p':0.220,'sig':False},
                              'beta':{'img':0.168,'exc':0.166,'p':0.129,'sig':False}},
}
ts={'clustering_coefficient':'Clustering Coefficient',
    'global_efficiency':'Global Efficiency',
    'bc_mean':'Betweenness Centrality',
    'modularity':'Modularity'}

fig,axes=plt.subplots(2,2,figsize=(16,12))
fig.patch.set_facecolor('white')
fig.suptitle("Statistical Analysis — Wilcoxon Signed-Rank Test\n"
             "105 Subjects | Motor Imagery vs Motor Execution",
             fontsize=14,fontweight='bold',color=C['text'])

for idx,(metric,title) in enumerate(ts.items()):
    ax=axes.flatten()[idx]
    for bi,band in enumerate(['mu','beta']):
        d=stat[metric][band]
        vals=[d['img'],d['exc']]
        bcols=[C['imagery'],C['execution']]
        x=[bi*3,bi*3+0.8]
        bars=ax.bar(x,vals,color=bcols,alpha=0.85,
                    edgecolor='white',linewidth=1.2,width=0.75)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.01,
                    f'{val:.4f}' if metric=='bc_mean' else f'{val:.3f}',
                    ha='center',fontsize=9,fontweight='bold')

        y_ann=max(vals)*1.08
        sig_text=f"p={d['p']:.3f}  {'★' if d['sig'] else 'ns'}"
        col_ann=C['accent'] if d['sig'] else '#888888'
        fw='bold' if d['sig'] else 'normal'
        fs=11 if d['sig'] else 9
        ax.text(bi*3+0.4,y_ann,sig_text,
                ha='center',fontsize=fs,color=col_ann,fontweight=fw)
        ax.text(bi*3+0.4,-max(vals)*0.13,
                'Mu Band' if band=='mu' else 'Beta Band',
                ha='center',fontsize=10,fontweight='bold',color=C['text'])

    ax.set_xticks([])
    ax.set_title(title,fontsize=11,fontweight='bold',pad=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylabel("Value",fontsize=10)
    ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.2)

    # Legend εκτός μπάρας — δεν κόβει τιμές   ← fix
    li=mpatches.Patch(color=C['imagery'],  label='Motor Imagery')
    le=mpatches.Patch(color=C['execution'], label='Motor Execution')
    ax.legend(handles=[li,le],fontsize=9,
              loc='upper right',bbox_to_anchor=(1.0,1.0))

plt.tight_layout(pad=3.0)
save("CH5_Fig5_statistical_summary")

# ════════════════════════════════════════════════════════════
# ΤΕΛΙΚΗ ΣΥΝΟΨΗ
# ════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("🎉 FINAL PLOTS v5 — ΟΛΟΚΛΗΡΩΘΗΚΑΝ!")
print("="*65)
print("\nΚεφάλαιο 4 — Μεθοδολογία:")
for f in ["FINAL_CH4_Fig1_PSD","FINAL_CH4_Fig2_epochs",
          "FINAL_CH4_Fig3_connectivity","FINAL_CH4_Fig4_brain_graphs",
          "FINAL_CH4_Fig5_metrics_S001"]:
    print(f"  ✅ {f}.png")
print("\nΚεφάλαιο 5 — Αποτελέσματα:")
for f in ["FINAL_CH5_Fig1_metrics_comparison","FINAL_CH5_Fig2_cc_mu_boxplot",
          "FINAL_CH5_Fig3_mu_vs_beta","FINAL_CH5_Fig4_correlation_heatmap",
          "FINAL_CH5_Fig5_statistical_summary"]:
    print(f"  ✅ {f}.png")
print("\n✅ Αυτά είναι τα ΤΕΛΙΚΑ γραφήματα για τη διπλωματική!")
print("="*65)
