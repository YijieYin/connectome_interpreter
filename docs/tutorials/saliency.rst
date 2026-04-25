Saliency
========

While :doc:`activation maximisation<act_max>` answers "what stimulus should I give to activate this neuron?", saliency answers the complementary question: **given a stimulus, which parts of it mattered most?**

Saliency is another technique from Explainable AI (`Simonyan et al. 2014 <https://arxiv.org/abs/1312.6034>`_). For a differentiable model, the importance of each input dimension to a target neuron's activation is simply the gradient of the target with respect to that input: how much does the target change when the input changes a little? Since our knowledge of the fruit fly brain is scattered across the nervous system, we extend this so gradients can be taken with respect to *any* neuron in the model, not only input neurons - see :py:func:`saliency` and :py:func:`get_gradients`.

Relation to signed effective connectivity 
------------------------------------------
In a purely linear system, the gradient of a target with respect to an input is exactly the :doc:`signed effective connectivity<ei_matmul>`. In the non-linear model the two come apart: the more non-linearity in play (saturated neurons, thresholding), the more the gradient depends on the current activation state, and the less the signed effective connectivity captures that state. Saliency therefore can vary based on the current stimulus and activation levels. Even with the non-linear :py:func:`MultilayeredNetwork`, if it operates largely in the linear range (where e.g. neurons barely saturate and/or where thresholding barely happens), then the results are more similar to signed effective connectivity, and more independent of the input. 