History
=======

Unreleased
----------
Behaviour changes in ``produce_tfrecords`` (rebuild TFRecords made with
earlier versions):

* Training records are written in random order. Before, they were written in
  input order and ``Dataset`` only shuffles within a buffer of 5, so with
  inputs sorted by the target each batch covered a narrow range of targets.
* With ``test_set='holdout'``, records keep their sample index ``'n'``. Before,
  the remaining records were renumbered while the folds kept the original
  indices, so part of the data dropped out of training and validation.
  Holdout records also carry their sample index.
* ``predefined_split`` is no longer modified in place. Its indices must lie
  within their input's samples (offset by the previous inputs' samples, as in
  ``MetaData.data['folds']``); before, other indices silently dropped samples
  from training and validation, now they raise a ``ValueError``.
* ``scale=True`` min-max scales each sample for ``input_type='fconn'``
  (new ``utils.minmax_scale``, which scales every non-zero entry) instead of
  scaling to a baseline.

Functional connectivity models:

* ``SymmetricModel`` reads ``attention_rate`` (as documented) instead of
  ``attention_ratio``.
* ``WeightedSum3dModel`` uses the scope ``weighted_sum_3d`` throughout (the
  metadata held ``model2_weighted_sum_3d``) and builds its graph only once.
* ``SymmetricModel.get_roi_labels`` can also return the stability of the
  patterns across folds (``return_stability=True``).
* ``extract_patterns`` and ``collect_patterns`` count only the trained CV
  folds, not the holdout fold.
* ``SymmetricModel.extract_patterns`` also works with
  ``trainable_pointwise_kernel=False`` (one frequency weighting shared by all
  components).
* The connectivity models are included in the API documentation.

Evaluation:

* Test sets are no longer shuffled, so test predictions are in the order of
  ``MetaData.data['test_fold']``. With several inputs, ``test_fold`` holds
  the holdout samples of all inputs (before, only those of the last input).
* ``train(..., store_test_predictions=True)`` keeps each fold model's
  predictions for the test set in ``cv_test_predictions`` (e.g. to average
  the fold models' predictions per sample), also with ``mode='loso'`` and
  subjects of different sizes.
* ``train`` passes ``shapley_order`` to ``collect_patterns`` only for models
  whose ``collect_patterns`` takes it (e.g. LFCNN), so the connectivity models
  can collect patterns.

Logging:

* ``update_log`` no longer fails for models without one of the logged specs
  (e.g. ``n_latent``), and logs the specs' ``model_id`` as ``spec_model_id``.
  A log written with other columns (e.g. by an earlier version) is rewritten
  with the union of the columns before appending, so all rows stay aligned.

0.7.0 (22-09-2026)
-------------------
* Migrated to standalone keras. Removed support for Python 3.9 and earlier
* Unit tests, Integration tests, and CI pipeline
* Major documentation updates

0.6.1 (22-05-2025)
-------------------
* Added Functional connectivity models in fc_models.py
* Moved MetaData class into a separate module meta.py
* Introduced novel interpretatbility tools and plots for lfcnn.py


0.2.0-beta (24-09-2020)
-----------------------
* updated API and documentation for compatibility with Tensorflow 2.0
* added Implementations of FBCSP-ShallowNet and Deep4
* added sorting components for interpretation by recursive elimination to LFCNN



0.1.1-beta (24-06-2019)
-----------------------
* massively updated API and examples. 
* added basic documentation.
* compatibility with previous version broken.


0.1.0-alpha (20-05-2019)
------------------------
* initial version
