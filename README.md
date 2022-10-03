# FoSR
FoSR: First-order spectral rewiring for addressing oversquashing in GNNs

## Requirements
To configure and activate the conda environment for this repository, run
```
conda env create -f environment.yml
conda activate oversquashing
```

## Experiments
To run experiments for the TUDataset benchmark, run the file ```run_graph_classification.py```. The following command will run the benchmark for FoSR with 20 iterations:
```
python run_graph_classification.py --rewiring fosr --num_iterations 20
```
