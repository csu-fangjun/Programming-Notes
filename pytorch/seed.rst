seed
====

.. code-block::

   import numpy as np
   seed = 234

   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)

