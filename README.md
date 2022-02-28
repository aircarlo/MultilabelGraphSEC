# Multilabel_Graph_SEC


This repository contains the python implementation for the hybrid architecture (Convolutional-Recurrent NN + Gated Graph NN) described in the following paper: [_Exploiting Graph-structured class correlations for Multilabel Sound Event Classification_](http://)

The FSD50K dataset was described in:
[_FSD50K: an Open Dataset of Human-Labeled Sound Events (2020) - Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra_](https://arxiv.org/pdf/2010.00475.pdf)
and can be downloaded from [Zenodo](http://doi.org/10.5281/zenodo.4060432).

To reproduce the experiments, set the parameters in `params.cfg` according to your environment, then run `python main.py --mode train` for train, or `python main.py --mode test` for evaluation.
