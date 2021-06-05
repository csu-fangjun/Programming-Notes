GMM
===

gconsts
-------
Its size is ``num_mix``. Its purpose is to pre-compute some values
that depend only on mean and vars.

.. code-block::

  p(x) = \sum_{i=0}^{n-1} \lamda_i \frac{1}{(2 \pi)^(d/2) det(\Sigma)^(1/2) }\exp ((x - \mu)^T \Sigma^{-1} (x - \mu)/(-1/2))

  log(p(x)) = log_sum_exp of F

  F_i = \log (\lambda_i) - (d/2 * \log (2 \pi)) + 0.5 * log(\frac{1}{det(\Sigma)}) - 0.5 * (\mu * \mu * (\Sigma)^(-1))

split
-----

For example, the current number of component is 10. To split it to 12, it needs two splits.

Split once, it becomes 11 components. Split twice, it becomes 12 components. Each split
increments the number of components.

During each split, it select the component that has the largest weight to split.
After splitting, one component has mean `mean + pertub_factor*std_dev`, the other
component has mean `mean - perturb_factor * stddev`. The variance of the two
components are the same as the original one.

merge
-----

logdet is computed by `0.5 * log(inv_vars)`. The objf is
to maximize `(w_i + w_j) * 0.5 * log(inv_var_merged_ij) - w_i * 0.5 * log(inv_vars_i) - w_j * 0.5 * log(inv_vars_j)`.

To merge component i and j, its mean is `(w_i * mean_i + w_j * mean_j)/(w_i + w_j)`.
For the variance, it first computes `E(x_i^2)`.  Then it computes merged `E(x_merged^2)`,
which is `w_i * E(x_i^2) + w_j * E(x_j^2)`.

Merge weight is `w_ij = w_i + w_j`.

Note that only first order statistics and second order statistics can be linearly
combined.
