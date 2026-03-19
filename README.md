###### \# **NeuroGame Transformer**: This repository contains the official implementation of the "NeuroGame Transformer" (NGT)--A novel attention mechanism that integrates "Cooperative Game Theory" (Shapley values, Banzhaf indices) with "Statistical Physics" (mean-field Ising model) for interpretable natural language inference. NeuroGame Transformer has been developed by the following people: Djamel Bouchaffra (Paris-Saclay University, Campus UVSQ, France), Ykhlef Fayçal (CDTA, Algeria), Mustapha Lebbah (Paris-Saclay University, Campus UVSQ, France), Hanene Azzag (Paris-Sorbonne-Nord University, France), and Bilal Faye (Paris-Sorbonne-Nord University, France).



\## Results: NGT achieves approximatively 86.6% test accuracy on SNLI outperforming all efficient transformers baselines and approaching the performance of strong pretrained models. It achieves an accuracy close to 80% on the MNLI dataset.

This project uses the SNLI and MNLI datasets. You can download them from the official source:

SNLI https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
Code: NeuroGame-Transformer-SNLI (3 eopochs were used)

MNLI (GLUE version): https://dl.fbaipublicfiles.com/glue/data/MNLI.zip 
Code: NeuroGame-Transformer-MNLI (10 epochs were used)

After downloading, place the following CSV files in a folder (e.g., `data/`):



\- `snli\_train.csv`

\- `snli\_validation.csv`

\- `snli\_test.csv`

Then update the `base\_path` variable in the code to point to your data folder.the CSV files should have the columns premise, hypothesis, label (as your code expects). 

\## Key Features

\-Game Theoretic Attribution: Shapley values and Banzhaf indices estimated via importance-weighted Monte Carlo.


\-Mean-Field Ising Model: Tokens interaction modeled as spins with temperature gamma solved via fixed-point iterations.


\-Interpretability: Local fields and pairwise interactions provide token-level and pairwise explanations.


\-Scalability: Monte Carlo sampling with Gibbs weights avoids exponential complexity.


\## Configuration

The reported results were obtained with:

\-n\_spins = 128

\-gamma (Temperature)=0.25

\-T\_mf=25

\-K\_mc\_train = 15, K\_mc\_eval=25

\-Learning rate = 3e(-5), MultiStepLR scheduler (drops after epochs 3 and 4)

\-Dropout 0.15, label smoothing 0.1, mixup alpha=0.2, EMA decay 0.999



\##How to Run

1. Clone the repository:

Bash

git clone....

