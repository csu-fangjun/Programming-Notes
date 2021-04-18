module
======

See ``pytorch/torch/nn/moduels/module.py``.

.. CAUTION::

  ``module.to()`, ``module.float()``, etc, changes the given module **in-place**,
  even if the method name does not end with an underscore.

Its has the following attributes:

  - ``self.training = True``
  - ``self._parameters = OrderedDict()``
  - ``self._buffers = OrderedDict()``
  - ``self._non_persistent_buffers_set = set()``
  - ``self._bacward_hooks = OrderedDict()``
  - ``self._forward_hooks = OrderedDict()``
  - ``self._forward_pre_hooks = OrderedDict()``
  - ``self._state_dict_hooks = OrderedDict()``
  - ``self._load_state_dict_pre_hooks = OrderedDict()``
  - ``self._moduels = OrderedDict()``

Some methods

- ``self.register_buffer(name, value, is_persistent)``.
  It updates ``self._buffers[name] = value``. If ``is_persistent``
  is True, a copy of name is saved in ``self._non_persistent_buffers_set``.

  Unlike parameters, buffers usually do not need autograd.

  It is perfectly to use ``self.register_buffer('some_name', None)``

- ``self.register_parameters(name, value)``. It is saved in ``self._parameters``.
  ``value`` can either be None or an instance ``nn.Parameter``.
  ``name`` and ``value`` are saved in ``self._parameters``.

  Usually, ``value`` is an ``nn.Parameter`` that ``requires_grad``.
  ``value`` has to be a leaf tensor if it needs gradient.


- ``self.add_module(name, value)``. ``value`` must be of type ``nn.Module``.
  It updates ``self._modules``


- ``self.get_submodule(name)`` can be used to get a module by its name,
  e.g., ``model.get_submodule('net1.linear')``. Note that it is more efficient
  than ``self.named_modules()``.

- ``self.get_parameter(name)`` can be used to get a parameter,
  e.g., ``model.get_parameter('net1.linear.weight')``. It checks that
  the module ``net1.linear`` exists and its return value is a ``nn.Parameter``.
  It uses ``self.get_submodule`` to find the submodule.

- ``self.get_buffer(name)``, similar to ``self.get_parameter(name)``.

Example 1
---------

Test `init`.

.. literalinclude:: ./code/module_test/ex1.py
  :caption: code/module_test/ex1.py
  :language: python
  :linenos:

Output::

  Parameter containing:
  tensor([[-0.1078]], requires_grad=True)
  Parameter containing:
  tensor([[0.]], requires_grad=True)
  tensor([1])
