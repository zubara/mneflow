# MNEflow
Neural networks for EEG-MEG decoding with MNE-python and Tensorflow.

# Installation 

```
pip install mneflow
```

## Dependencies
[MNE-Python](https://mne.tools/stable/index.html)>=0.14
[Tensorflow](https://www.tensorflow.org/) >=1.12, >=2.1
[NumPy](https://numpy.org/)
[Scikit-Learn](https://scikit-learn.org/stable/)

# Documentation

API reference is avalable in the [Documentation](https://mneflow.readthedocs.io/en/latest/).

# Examples

Also check the [example notebooks](https://github.com/zubara/mneflow/blob/master/examples/mneflow_example_tf2.ipynb).

# References 
When using the implemented models please cite: 

###  for LF-CNN or VAR-CNN 
Zubarev I, Zetter R, Halme HL, Parkkonen L. Adaptive neural network classifier for decoding MEG signals. Neuroimage. 2019 May 4;197:425-434. [link](https://www.sciencedirect.com/science/article/pii/S1053811919303544?via%3Dihub)

```
@article{Zubarev2019AdaptiveSignals.,
    title = {{Adaptive neural network classifier for decoding MEG signals.}},
    year = {2019},
    journal = {NeuroImage},
    author = {Zubarev, Ivan and Zetter, Rasmus and Halme, Hanna-Leena and Parkkonen, Lauri},
    month = {5},
    pages = {425--434},
    volume = {197},
    url = {https://linkinghub.elsevier.com/retrieve/pii/S1053811919303544 http://www.ncbi.nlm.nih.gov/pubmed/31059799},
    doi = {10.1016/j.neuroimage.2019.04.068},
    issn = {1095-9572},
    pmid = {31059799},
    keywords = {Brain–computer interface, Convolutional neural network, Magnetoencephalography}
}
```

### for EEGNet 
```
@article{Lawhern2018,
  author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
  title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
  year={2018}
}
```

### for Deep-4 and FBCSP-ShallowNet
```
@article{Schirrmeister2017DeepVisualization,
    title = {{Deep learning with convolutional neural networks for EEG decoding and visualization}},
    year = {2017},
    journal = {Human Brain Mapping},
    author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
    number = {11},
    month = {11},
    pages = {5391--5420},
    volume = {38},
    url = {http://doi.wiley.com/10.1002/hbm.23730},
    doi = {10.1002/hbm.23730},
    issn = {10659471},
    keywords = {EEG analysis, brain, brain mapping, computer interface, electroencephalography, end‐to‐end learning, machine interface, machine learning, model interpretability}
}
```
