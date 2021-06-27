Container
=========

Defined in ``torch/nn/modules/container.py``.


nn.Sequential
============

.. code-block::

   # Example of using Sequential
   model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
          )

It uses:

.. code-block:: python
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args: Any):
       '''some code '''

Note that we can use overload to signify that ``__init__`` accepts
multiple kinds of values.

``nn.Sequential`` also accepts an instance of ``OrderedDict``. The difference
is that ``OrderedDict`` can assign a name to each module, while the name of
a module in a list is its index.

nn.ModuleList
=============

Note that it does not have ``forward()``.

.. code-block:: python

   class MyModule(nn.Module):
      def __init__(self):
          super(MyModule, self).__init__()
          self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

      def forward(self, x):
          # ModuleList can act as an iterable, or be indexed using ints
          for i, l in enumerate(self.linears):
              x = self.linears[i // 2](x) + l(x)
          return x


nn.ModuleDict
=============

Note that it does not have ``forward()``.

.. code-block::

   class MyModule(nn.Module):
      def __init__(self):
          super(MyModule, self).__init__()
          self.choices = nn.ModuleDict({
                  'conv': nn.Conv2d(10, 10, 3),
                  'pool': nn.MaxPool2d(3)
          })
          self.activations = nn.ModuleDict([
                  ['lrelu', nn.LeakyReLU()],
                  ['prelu', nn.PReLU()]
          ])

      def forward(self, x, choice, act):
          x = self.choices[choice](x)
          x = self.activations[act](x)
          return x
