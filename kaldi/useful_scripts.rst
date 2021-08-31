Useful scripts
==============

.. code-block::

  cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/tokenid.scp

where ``token.scp`` contains something like::

  foo a b c
  bar d e f

and ``dic`` contains something like::

  <eps> 0
  <unk> 1
  a 2
  b 3
  ...

and ``oov`` is ``<unk>``


.. code-block::

  vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
  odim=`echo "$vocsize + 2" | bc`
  awk -v odim=${odim} '{print $1 " " odim}' ${dir}/text > ${tmpdir}/odim.scp

If the last line of ``dic`` is ``a 100``, then
``vocsize`` is 100, ``odim`` is 102. ``odim.scp`` contains something like::

  foo 102
  bar 102
