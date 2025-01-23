==========================================================
Data Repository
==========================================================
Source code and data files for the manuscript. Execute plot.ipynb to view the data.

How to cite
-----------
If this data is used, please cite W. He, J. Sears, F. Barantani, T. Kim, J. W. Villanova, T. Berlijn, M. Lajer, M. A.
McGuire, J. Pelliciari, V. Bisogni, S. Johnston, E. Baldini, M. Mitrano, and M. P. M. Dean, accepted in Phys. Rev. X (2025)

Run locally
-----------

Work with this by installing `docker <https://www.docker.com/>`_ and pip and then running

.. code-block:: bash

       pip install jupyter-repo2docker
       jupyter-repo2docker  --editable --Repo2Docker.platform=linux/amd64 .

Change `tree` to `lab` in the URL for JupyterLab.

Run remotely
------------

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/mpmdean/He2024dispersive/HEAD?filepath=plot.ipynb


