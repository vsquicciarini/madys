Installation
=====

.. _installation:

Install
------------

As catalog queries are mediated by the `TAP Gaia Query <https://github.com/mfouesneau/tap>`_ package (``TAP``), this module needs being present in your working environment. If you are trying to import ``MADYS`` from the command line, the module is automatically installed if not found. However, **this does not work on Jupyter Notebook**; in this case, we suggest to manually install the package via pip, through the following command:

.. code-block:: console

  (.venv) $ pip install git+https://github.com/mfouesneau/tap
   
Please make sure **not** to write just `pip install tap`; this command would download **a different**, although eponymous, package. 

.. note::

   ``TAP`` requires the installation of `lxml <https://lxml.de/>`_. If an error is thrown during the import of ``lxml``, cancel the ``lxml`` package from your working path and force pip to install the version 4.6.3, whose compatibility with ``TAP`` has been extensively tested.

After installing ``TAP``, the installation of ``MADYS`` can be smoothly performed through pip:

.. code-block:: console

   (.venv) $ pip install madys

Dependencies
----------------

This package relies on usual packages for data science and astronomy: numpy (v1.18.1), scipy (v1.6.1), pandas (v1.1.4), matplotlib (v3.3.4), astropy (v4.3.1) and h5py (v3.2.1). In addition, it also requires astroquery (v0.4.2.dev0).

As mentioned above, two additional dependencies are represented by TAP Gaia Query (v0.1) and lxml (v4.6.3).
