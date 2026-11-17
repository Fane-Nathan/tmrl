Command Line Interface
======================

``tmrl`` provides commands for users who wish to use the readily implemented example pipelines for TrackMania.

Examples:
---------

Initialize the local ``TmrlData`` folder and default configuration:

.. code-block:: bash

   python -m tmrl --install

Launch the default training pipeline for TrackMania on 3 possibly different machines:

.. code-block:: bash

   python -m tmrl --server
   python -m tmrl --trainer
   python -m tmrl --worker

Test (deploy) the readily trained example policy for TrackMania:

.. code-block:: bash

   python -m tmrl --test

Launch the reward recorder in your own track in TrackMania:

.. code-block:: bash

   python -m tmrl --record-reward

Check that the TrackMania environment is working as expected:

.. code-block:: bash

   python -m tmrl --check-environment

Benchmark the RolloutWorker in TrackMania (requires ``"benchmark": true`` in ``config.json``):

.. code-block:: bash

   python -m tmrl --benchmark

Launch the Trainer and enable logging to wandb.ai:

.. code-block:: bash

   python -m tmrl --trainer --wandb
