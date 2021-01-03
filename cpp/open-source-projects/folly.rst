Folly
=====

FBString
--------

.. code-block::

  enum class Category {
    isSmall = 0,
    isMedium = 0x80,
    isLarge = 0x40,
  };

  struct MeidumLarge {
    Char* data_;
    size_t size_;
    size_t capacity_;
  };

  union {
    uint8_t bytes_[sizeof(MediumLarge)];
    Char small_[sizeof(MediumLarge) / sizeof(Char)];
    MediumLarge ml_;
  };

- For 32-bit:

  - `kCategoryShift` is 24
  - `capacityExtractMask` is: ~(0xc0 << 24)
  - `sizeof(MediumLarge)` is 12 byte
  - `lastChar` is 11

- For 64-bit

  - `kCategoryShift` is 56
  - `capacityExtractMask` is: ~(0xc0 << 56)
  - `sizeof(MediumLarge)` is 24 bytes
  - `lastChar` is 23


