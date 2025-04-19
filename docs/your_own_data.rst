Using your own dataset
======================

Essentially, you need the following two components: 

- a connectivity matrix in sparse form (pre in rows, post in columns), and 
- a dataframe containing meta-information (e.g. neuron id, cell type etc.) corresponding to the indices in the connectivity matrix.

For examples for making these, see the ipython notebooks with "prepare_connectome" in the file name, `here <https://github.com/YijieYin/interpret_connectome>`_.

Values in the connectivity matrix 
---------------------------------
So far, I have been using input proportion. For instance, the connection weight from A to B is: 

.. math::
    \frac{\text{the number of synapses from A to B}}{\text{the total number of post-synapses B has}}

This means that, except for incoming neurons with no input in the brain, the sum across rows for each column (postsynaptic neuron) is 1. See more discussion on this in the :doc:`effective connectivity calculation tutorial <tutorials/matmul>`.

**Note:** calculating the effectivity connectivity is essentially getting all the paths of a certain length from source to target neurons, and summing the weights across the paths. This means that if the data you have is partially reconstructed, or if you are only taking a part of the connectome, you should be cautious, and if possible, use *all* post-synapses a neuron has as the denominator.

Why does it need to be sparse? 
--------------------------------
Storing big, dense matrices can be memory-intensive. For example, if there are 50,000 neurons, a dense matrix would require: 

.. math::
    50,000^2 \times 32 \text{ bits} / 8 \text{ bits per byte} = 10^{10} \text{ bytes} = 10 \text{ GB}.

However, if the connectivity matrix is sparse, i.e. only a small fraction of the entries are non-zero, the same connectivity information is better stored as an edgelist (pre, post and connection weight). This is essentially what's happening when you use the `scipy.sparse.coo_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix>`_ format. 

I'm using `float32` (i.e. can be a decimal number, needs 32 bits to represent) to represent the connection weights, to get a balance between memory usage and precision (other options include `float8`, `float64` etc.).
