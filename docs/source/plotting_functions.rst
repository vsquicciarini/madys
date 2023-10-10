Plotting functions
=====


Extinction
------------

The function :py:func:`SampleObject.plot_2D_ext` can be used to plot the integrated extinction along the line of sight, given as input a certain field of view. As a static method, it does not require the creation of a SampleObject instance.

Let us compute the integrated extinction expected at the coordinates and distance of the Upper Scorpius association, using the 3D extinction map by Leike et al. (2020).

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from madys import *
>>> 
>>> SampleObject.plot_2D_ext(ra=[236,252], dec=[-30,-16], d=160, color='G',reverse_xaxis=True,cmap='gray_r',fontsize=18)

The command will produce the following figure:
.. image:: images/example_ext_map2.png

It is possible, through the keyword ``ax``, to overplot the extinction map over an existing figure. Let us try, for instance, to plot some random points over the abovementioned extinction map:

>>> fontsize = 20
>>> fig, ax = plt.subplots(figsize=(8,8))
>>> SampleObject.plot_2D_ext(ra=[236,252],dec=[-30,-16],d=160,color='G',reverse_xaxis=True,cmap='gray_r',fontsize=fontsize,colorbar=False,ax=ax)
>>> ax.xaxis.set_tick_params(labelsize=fontsize)
>>> ax.yaxis.set_tick_params(labelsize=fontsize)
>>> ax.plot(240+5*np.random.rand(10),-25+5*np.random.rand(10), marker = '*', linestyle = '', color = 'red')
>>> plt.show()

.. image:: images/example_ext_map.png


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
