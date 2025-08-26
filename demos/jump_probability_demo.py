import csv
import math
import numpy as np
import networkx as nx
import snafu
import os
from pathlib import Path
import csv

out_dir = os.path.join(os.path.dirname(__file__), "demos_data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "jump_probability_results.csv")

G = nx.Graph()
with open('demos_data/madrid_network.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  
    for row in reader:
        if not row or len(row) < 4:
            continue
        _, item1, item2, edge = row[:4]
        if str(edge).strip() == "1":
            G.add_edge(item1, item2)
        else:
            G.add_node(item1)
            G.add_node(item2)

items = {inum: node for inum, node in enumerate(list(G.nodes()))}
group_network = nx.to_numpy_array(G)

print("Network stats:",
      "nodes =", G.number_of_nodes(),
      "edges =", G.number_of_edges())

# ---------- Load and align fluency data ----------
patient_data = snafu.load_fluency_data("demos_data/PAFIP_animal.txt")

revItems = snafu.reverseDict(items)  # label -> index
fluency_data = []
for lst in patient_data.labeledlists:
    fluency_data.append([revItems[label] for label in lst if label in revItems])

# ---------- Data model  ----------
data_model = snafu.DataModel({
    'jump': 0.05,
    'jumptype': 'uniform',
    'start_node': 'uniform'
})

# ---------- Estimate best jump per subject/list ----------
best_jumps = []
log_likelihoods = []  
for sub_idx, fluency_list in enumerate(fluency_data):
    if not fluency_list:
        print(f"Subject {patient_data.subs[sub_idx]}: list empty after alignment — skipping.")
        best_jumps.append(np.nan)
        log_likelihoods.append(np.nan)  
        continue

    #best-jump estimation 
    best_jump = snafu.estimateJumpProbability(group_network, data_model, fluency_list)
    best_jumps.append(best_jump)

    # compute log-likelihood
    try:
        if isinstance(best_jump, (int, float)) and not np.isnan(best_jump):
            data_model.jump = float(best_jump)
            ll, p_by_transition = snafu.probX([fluency_list], group_network, data_model)
            log_likelihoods.append(ll)
        else:
            log_likelihoods.append(np.nan)
    except Exception:
        log_likelihoods.append(np.nan)

valid = [x for x in best_jumps if isinstance(x, (int, float)) and not math.isnan(x)]
DEMOS_DIR = Path(__file__).resolve().parent
DEMOS_DATA = DEMOS_DIR / "demos_data"
DEMOS_DATA.mkdir(parents=True, exist_ok=True)

out_csv = "demos_data/jump_probability_results.csv"  # (left as in your original)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id","best_jump_param","log_lik"])

    for i, bj in enumerate(best_jumps):
        subj = patient_data.subs[i]
        ll = log_likelihoods[i]
        # write empty cell for NaN
        bj_cell = "" if (isinstance(bj, float) and math.isnan(bj)) else bj
        ll_cell = "" if (not isinstance(ll, (int, float)) or math.isnan(ll)) else ll
        w.writerow([subj, bj_cell, ll_cell])

print(f"Saved results to: {out_csv}")


     


