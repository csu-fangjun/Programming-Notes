
Context Free Grammer
====================

Backus-Naur Form (BNF):

.. code-block::

  <name>::=         tom | dick | harray
  <sentence>::=     <name> | <list> and <name>
  <list>::=         <name>, <list>  | <name>
