Implemented models
******************

LFCNN
-----

.. autoclass:: mneflow.lfcnn.LFCNN
    :members:
    :undoc-members:
    :show-inheritance:

VARCNN
------

.. autoclass:: mneflow.models.VARCNN
    :members:
    :undoc-members:
    :show-inheritance:

EEGNet
------

.. autoclass:: mneflow.models.EEGNet
    :members:
    :undoc-members:
    :show-inheritance:

FBCSP_ShallowNet
----------------

.. autoclass:: mneflow.models.FBCSP_ShallowNet
    :members:
    :undoc-members:
    :show-inheritance:

Deep4
-----

.. autoclass:: mneflow.models.Deep4
    :members:
    :undoc-members:
    :show-inheritance:


Functional connectivity models
------------------------------

Models for functional connectivity inputs (``input_type='fconn'`` in
:func:`mneflow.produce_tfrecords`), with inputs of shape
``(n_nodes, n_nodes, n_frequencies)``.

Conv3DModel
~~~~~~~~~~~

.. autoclass:: mneflow.fc_models.Conv3DModel
    :members:
    :undoc-members:
    :show-inheritance:

WeightedSum3dModel
~~~~~~~~~~~~~~~~~~

.. autoclass:: mneflow.fc_models.WeightedSum3dModel
    :members:
    :undoc-members:
    :show-inheritance:

SymmetricModel
~~~~~~~~~~~~~~

.. autoclass:: mneflow.fc_models.SymmetricModel
    :members:
    :undoc-members:
    :show-inheritance:
