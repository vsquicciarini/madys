Plotting functions
=====


Extinction
------------

The function :py:func:`SampleObject.plot_2D_ext` can be used to plot the integrated extinction along the line of sight, given as input a certain field of view. As a static method, it does not require the creation of a SampleObject instance.


>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from madys import *
>>> 
>>> ra_range, dec_range, distance = [236, 252], [-30, -16], 160
>>> grid_x=np.linspace(236,252,1000)
>>> grid_y=np.linspace(-30,-16,1000)
>>> XX, YY = np.meshgrid(grid_x,grid_y)
>>> fontsize=20
>>> fig, ax = plt.subplots(figsize=(8,8))
>>> SampleObject.plot_2D_ext(ra=[236,252], dec=[-30,-16], d=160, color='G',reverse_xaxis=True,cmap='gray_r',fontsize=fontsize,colorbar=False,ax=ax)
>>> ax.xaxis.set_tick_params(labelsize=fontsize)
>>> ax.yaxis.set_tick_params(labelsize=fontsize)
>>> ax.yaxis.set_tick_params(which='minor', bottom=False)
>>> ax.yaxis.set_tick_params(which='major', bottom=False)
>>> ax.set_xlabel(r'$\alpha [^\circ]$',fontsize=fontsize)
>>> plt.show()


.. image:: images/example_ext_map.png


.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
