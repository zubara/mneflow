# MNEflow

[![Tests](https://github.com/zubara/mneflow/actions/workflows/tests.yml/badge.svg)](https://github.com/zubara/mneflow/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/mneflow/badge/?version=latest)](https://mneflow.readthedocs.io/en/latest/?badge=latest)

Neural networks for EEG/MEG decoding and interpretation, built on [MNE-Python](https://mne.tools) and [TensorFlow](https://www.tensorflow.org/).

Full documentation: **[mneflow.readthedocs.io](https://mneflow.readthedocs.io/)**

MNEflow provides neuroscientists with a robust, reproducible, and time-efficient way to apply deep neural networks (DNNs) to EEG and MEG data. It implements several popular architectures for M/EEG decoding, a streamlined pipeline for preprocessing, training, and benchmarking them, and a growing set of tools for inspecting the patterns a trained model has learned to rely on.

## Installation

```
pip install mneflow
```

## Documentation

API reference is available in the [Documentation](https://mneflow.readthedocs.io/en/latest/).

## Dependencies

| Component | Tested range | Why the bound is there |
| --- | --- | --- |
| Python | 3.10 – 3.13 | Use mneflow 0.6.1 for Python 3.9 and Earlier;|
| MNE-Python | >=1.10, <=1.13.2 | Use mneflow 0.6.1 with older veriosns of MNE;|
| TensorFlow | >=2.16.1, <=2.21.0 | tensorflow >= 2.16rc in mneflow 0.6.1 |
| Keras | >=3.0, <4 | mneflow 0.7.0 Migrated to standalone keras |
| NumPy / SciPy | no explicit ceiling | TensorFlow and MNE-Python each pin their own NumPy/SciPy ceiling per release |


See [`pyproject.toml`](pyproject.toml) for the exact, currently enforced version constraints.

## Software architecture

The functionality of MNEflow is organized around two blocks, mirroring the everyday workflow of a decoding study:

- **Preprocessing** — converts EEG/MEG data into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files, applying filtering, scaling, resampling, channel selection, and partitioning into training/validation/test folds. This step also handles machine-learning-specific transformations such as segmenting continuous recordings, producing sequences for recurrent/sequence models, augmenting the data, and transforming target variables.
- **Experimentation** — covers model design, training, hyperparameter optimization, logging, and interpretation.

Storing the preprocessed data and metadata on disk (rather than recomputing them for every run) avoids unnecessary repetition of preprocessing, keeps memory usage low, and lets different models be trained and benchmarked on exactly the same data partitions. A trained model is likewise saved to disk and can be reloaded, applied to a new dataset, or used to inspect the patterns behind its predictions.

## Functionality

**Import.** EEG/MEG data can be imported directly from MNE-Python by passing an `mne.Epochs` object to `mneflow.produce_tfrecords`. Data exported from other signal-processing software can be provided as a NumPy array with shape `[trials, sensors, time points]`, or as paths to `.fif`, `.mat`, or `.npz` files.

**Preprocessing.** `mneflow.produce_tfrecords` builds the TFRecords dataset and its accompanying `mneflow.MetaData`, and exposes the basic preprocessing utilities (filtering, scaling, channel selection, resampling) as well as machine-learning-specific ones. The `input_type` argument controls how each input is treated:

- `'trials'` — each input is an i.i.d. sample, producing a dataset of shape `(n, 1, t, ch)`;
- `'seq'` — each input is a sequence of shorter segments, for sequence/RNN-type models;
- `'continuous'` — inputs are treated as one continuous recording and segmented with a configurable stride (augmentation);
- `'fconn'` — inputs are treated as functional-connectivity data.

The `target_type` argument similarly distinguishes classification (`'int'`), regression of a scalar variable (`'float'`), and regression or classification against a continuous, possibly multichannel signal (`'signal'`, e.g. reconstructing a continuous source-level or envelope signal), the latter via a user-supplied `transform_targets` function.

**Model development.** MNEflow implements several published CNN architectures for EEG/MEG decoding, all inheriting from the same parent class (`mneflow.models.BaseModel`) and sharing the same datasets, optimizers, and validation routines, which makes them directly comparable and easy to benchmark against one another:

| Model | Reference |
| --- | --- |
| `LFCNN` | Zubarev et al. (2019), *NeuroImage* — [link](https://www.sciencedirect.com/science/article/pii/S1053811919303544) |
| `VARCNN` | Zubarev et al. (2019), *NeuroImage* — [link](https://www.sciencedirect.com/science/article/pii/S1053811919303544) |
| `EEGNet` | Lawhern et al. (2018), *J. Neural Eng.* — [link](http://stacks.iop.org/1741-2552/15/i=5/a=056013) |
| `FBCSP_ShallowNet` | Schirrmeister et al. (2017), *Human Brain Mapping* — [link](http://dx.doi.org/10.1002/hbm.23730) |
| `Deep4` | Schirrmeister et al. (2017), *Human Brain Mapping* — [link](http://dx.doi.org/10.1002/hbm.23730) |
| `SymmetricModel` | Ruuskanen, Saarro et al. (2026), *preprint* — [link](https://www.biorxiv.org/content/10.64898/2026.08.20.745932v1) |
| `WeightedSum3dModel` | Ruuskanen, Saarro et al. (2026), *preprint* — [link](https://www.biorxiv.org/content/10.64898/2026.08.20.745932v1) |
| `Conv3DModel` | Ruuskanen, Saarro et al. (2026), *preprint* — [link](https://www.biorxiv.org/content/10.64898/2026.08.20.745932v1) |

The modular structure of the underlying `mneflow.layers` also makes it straightforward to define a custom architecture by combining existing layers or adding new ones — see the [custom-network example](https://github.com/zubara/mneflow/blob/master/examples/own_graph_example.ipynb).

**Training and evaluation.** Training and evaluation are configured through `model.build()`: the optimizer, objective function, and performance metrics. All models are trained with Adam by default, using categorical cross-entropy for classification and mean-squared error for regression, with early stopping. `mneflow.losses` additionally provides objective functions for continuous multichannel targets, combining cosine-similarity, MSE/MAE, spectral (FFT-based), and Riemannian-distance terms. Training runs, logs, and trained models are kept on disk, making it easy to reproduce, inspect, and compare results across runs.

**Model inspection.** MNEflow provides tools to inspect the patterns a model has learned to rely on when making its predictions. At present, this is available for the `LFCNN` family of models, which impose a conditional-independence assumption on the latent components learned from the data: the input is decomposed into a small number of spatial (de-mixing) and temporal (convolution kernel) filter pairs, each treated as a conditionally-independent latent component, which makes their spatio-temporal properties directly interpretable. `model.compute_patterns()` computes spatial and temporal patterns, weights, spectra, and feature-relevance metrics for each latent component; `model.plot_topos()`, `model.plot_waveforms()`, and `model.plot_combined_pattern()` visualize them. Component relevance can be ranked by several complementary methods:

- **Weight-based contributions** — feature relevance ranked directly by the magnitude of the component's weights (e.g. the ℓ2 norm).
- **Correlation with the target variable** — feature relevance ranked by (absolute) Spearman correlation with the target for regression, or by categorical cross-entropy for classification.
- **Component-interaction (Shapley-like) relevances** — `shapley_order` in `model.compute_patterns()` controls whether single-component (order 1), pairwise (order 2), or triple-wise (order 3) interactions between latent components are evaluated for their effect on the loss, extending the single-component recursive-elimination approach to higher-order component interactions.

`mneflow.MetaData.get_feature_relevances()`, `get_spatial_patterns()`, and `get_spectra()` provide programmatic access to the computed patterns for further analysis.

## Publications using MNEflow

A selection of peer-reviewed papers and preprints that use MNEflow for EEG/MEG decoding or interpretation (compiled from [Google Scholar](https://scholar.google.com/citations?user=xWRyzr4AAAAJ); not necessarily exhaustive):

- Zubarev I, Nurminen M, Parkkonen L. Robust discrimination of multiple naturalistic same-hand movements from MEG signals with convolutional neural networks. *Imaging Neuroscience* 2, imag-2-00178 (2024). [link](https://doi.org/10.1162/imag_a_00178)
- Ruuskanen S, Saarro E, Caivano CM, Parkkonen L, Zubarev I. Interpretable Decoding of Frequency-Resolved Functional Connectivity. *bioRxiv* (2026). [link](https://www.biorxiv.org/content/10.64898/2026.08.20.745932v1)
- Matsuda RH, Makkonen M, Zubarev I, Kahilakoski OP, Kinnunen LA, et al. Automated robotic control system for EEG-BCI-guided closed-loop TMS. *bioRxiv* (2026). [link](https://www.biorxiv.org/content/10.64898/2026.05.15.725366v1)
- Pultsina K, Zubarev I, Ronkainen P, Parviainen T. Cortical Oscillatory Dynamics Track Sympathetic Arousal and Index Individual Differences in Anxiety. Preprint (2026). [link](https://doi.org/10.21203/rs.3.rs-10208090/v1)

## References

Zubarev I, Vranou G, Parkkonen L. MNEflow: Neural networks for EEG/MEG decoding and interpretation. *SoftwareX* [link](https://www.sciencedirect.com/science/article/pii/S2352711021001795)

When using the implemented models, please also cite the papers describing them:

### for LF-CNN or VAR-CNN

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

### for FBCSP-ShallowNet and Deep4

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

## License

BSD-3. See [LICENSE.md](LICENSE.md).
