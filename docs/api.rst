API reference
=============

The reference below is generated from the Python objects themselves. Sphinx
autodoc displays each object's current signature, annotations, and docstring;
autosummary creates a dedicated page for each listed API.

To add an API, add its fully qualified import path to the appropriate list.
For example, add ``cks_optimizer.cnn.some_function`` to the CNN workflow list.

CNN workflow
------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   cks_optimizer.cnn.build_optimize_config
   cks_optimizer.cnn.load_cnn_model
   cks_optimizer.cnn.data_process
   cks_optimizer.cnn.optimize_cnn_model
   cks_optimizer.cnn.export_onnx
   cks_optimizer.cnn.evaluate
   cks_optimizer.cnn.run_cnn_pipeline

Configuration
-------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   cks_optimizer.cnn.OptimizeConfig
   cks_optimizer.cnn.CNNConfig
   cks_optimizer.cnn.QuantConfig
   cks_optimizer.cnn.QuantizerSettings
   cks_optimizer.cnn.SparsityConfig
   cks_optimizer.cnn.CalibrationConfig
   cks_optimizer.cnn.ExportConfig

Dataset processing
------------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   cks_optimizer.cnn.ProcessedData
   cks_optimizer.cnn.ImagenetDataloader
   cks_optimizer.cnn.CocoDataloader
   cks_optimizer.cnn.VocDataloader
   cks_optimizer.cnn.normalize_dataset
   cks_optimizer.cnn.datasets.dataloaders.get_dataloader

Optimization, results, and evaluation
-------------------------------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   cks_optimizer.cnn.CNNOptimizer
   cks_optimizer.cnn.OptimizationResult
   cks_optimizer.cnn.EvaluationResult

Large-model optimization and CLI
--------------------------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   cks_optimizer.lm.LMOptimizer
   cks_optimizer.cli.main
