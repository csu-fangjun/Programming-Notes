unicode
=======

``U+0420`` is a codepoint. It has nothing to do with
the number in memory. The number in memory depends on
how the codepoint is encoded. Note that ``0420`` is a hex
number.

In python,

.. code-block::

  # 16-bits. Use lowercase u
  print('\u0394') # prints: Δ

  # 32-bits, Note that it is uppercase U
  print('\U0000000394') # prints: Δ

  a = '\xce\x94'
  assert isinstance(a, bytes)
  b = a.decode()
  assert isinstance(b, str)
  print(b) # prints: Δ

  c = b.encode()
  assert isinstance(c, bytes)
  assert c == '\xce\x94'

  print(bin(0x394)    # 1110_010100
  print(bin(0xce94))  # 11001110_10010100

  '\u00df' # ß
  '\u00df'.casefold() # ss


Same character with different code points:

.. code-block::

  '\u00ea'  # ê
  '\u0065\u0302'  # ê

  assert len('\u00ea') == 1
  assert len('\u0065\u0302') == 2
  assert '\u00ea' != '\u0064\u0302'

  '\u00ea'.encode() # b'\xc3\xaa'
  '\u0065\u0302'.encode()  # b'e\xcc\x82'
  '\u0065'.encode()  # b'e'
  '\u0302'.encode() # b'\xcc\x82'

  import unicodedata

  # compatible decomposition, then canonical decomposition
  unicodedata.normalize('NFKC', '\u00ea').encode() # b'\xc3\xaa'
  unicodedata.normalize('NFKC', '\u0065\u0302').encode() # b'\xc3\xaa'

  # canonical decomposition
  unicodedata.normalize('NFD', '\u00ea').encode() # b'e\xcc\x82'
  unicodedata.normalize('NFD', '\u0065\u0302').encode() # b'e\xcc\x82'



References
----------

- UTF-8 history `<https://www.cl.cam.ac.uk/~mgk25/ucs/utf-8-history.txt>`_

    It describes the story after UTF-8. There is also an C implementation


- The Absolute Minimum Every Software Developer Absolutely, Positively Must
  Know About Unicode and Character Sets (No Excuses!)

    `<https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/>`_
