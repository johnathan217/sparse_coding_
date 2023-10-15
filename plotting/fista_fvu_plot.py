import itertools

import standard_metrics
from autoencoders.fista import FunctionalFista
from fvu_sparsity_plot import score_dict, generate_scores, plot_scores
import os

# os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
# load dataset and dict
files = [
    ("SAE_ensemble", "D:/sparse_coding_/output_basic_test/normal_13_10_2023_iterative_set/learned_dicts_epoch_0.pt"),
    ("Fista", "D:/sparse_coding_/output_basic_test/fista_13_10_2023_iterative_set/learned_dicts_epoch_0.pt"),
    # ("Fista_iterative", "D:/sparse_coding_/output_basic_test/fista10_10_2023_iterative/learned_dicts_epoch_0.pt")
    ]
dataset_file = "D:/sparse_coding_/activation_data/layer_2/6.pt"

scores = generate_scores(files, dataset_file, device="cuda")
print(scores)
# plot graph
colors = ["Purples", "Blues", "Greens", "Oranges"]
styles = ["x", "+", ".", "*"]

settings = {
    label: {"style": style, "color": color, "points": True}
    for (style, color), label in zip(itertools.product(styles, colors), scores.keys())
}
filename = "fistavnormal"
#def plot_scores(scores, settings, xlabel, ylabel, xrange, yrange, title, filename):
plot_scores(
    scores,
    settings,
    "sparsity",
    "top-fvu",
    (0, 512),
    (0, 1),
    "Fista vs Normal",
    f"D:/sparse_coding_/output_basic_test/graphs/{filename}",
)