import itertools
import torch
import standard_metrics
from autoencoders.fista import FunctionalFista
from fvu_sparsity_plot import score_dict, generate_scores, plot_scores
import os


# load dataset and dict
files = [
    ("SAE_ensemble", "D:/sparse_coding_/output_basic_test/normal_13_10_2023_iterative_set/learned_dicts_epoch_0.pt"),
    ("Fista", "D:/sparse_coding_/output_basic_test/fista_13_10_2023_iterative_set/learned_dicts_epoch_0.pt"),
    # ("Fista_iterative", "D:/sparse_coding_/output_basic_test/fista10_10_2023_iterative/learned_dicts_epoch_0.pt")
    ("Fista300","D:/sparse_coding_/output_basic_test/fista300steps_17_10_2023_iterative_set/learned_dicts_epoch_0_chunk_0.pt" ),
("Fista300","D:/sparse_coding_/output_basic_test/fista300steps_17_10_2023_iterative_set/learned_dicts_epoch_0_chunk_7.pt" ),

    ]
dataset_file = "D:/sparse_coding_/activation_data/layer_2/6.pt"

scores = generate_scores(files, dataset_file, device="cuda")
print(scores) # prints as [ sparsity, fvu, c]
# plot graph
colors = ["Purples", "Blues", "Greens", "Oranges"]
styles = ["x", "+", ".", "*"]

settings = {
    label: {"style": style, "color": color, "points": True}
    for (style, color), label in zip(itertools.product(styles, colors), scores.keys())
}
filename = "fistavnormalvfista300"
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

