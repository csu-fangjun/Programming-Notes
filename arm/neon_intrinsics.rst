
Frequently used NEON Intrinsics
===============================



dup
---

- ``int32x4_t vdupq_n_s32 (int32_t value)``
- ``int32x2_t vdup_n_s32 (int32_t value)``
- ``int32x2_t vld1_dup_s32 (int32_t const * ptr)``
- ``int32x4_t vld1q_dup_s32 (int32_t const * ptr)``

mov
---

- ``int32x4_t vmovq_n_s32 (int32_t value)``
- ``int32x2_t vmov_n_s32 (int32_t value)``


ld
--

- ``int32x2_t vld1_s32 (int32_t const * ptr)``
- ``int32x4_t vld1q_s32 (int32_t const * ptr)``

st
--

- ``void vst1q_s32 (int32_t * ptr, int32x4_t val)``
- ``void vst1_s32 (int32_t * ptr, int32x2_t val)``
- ``void vst1_lane_s32 (int32_t * ptr, int32x2_t val, const int lane)``
- ``void vst1q_lane_s32 (int32_t * ptr, int32x4_t val, const int lane)``

add
---

- ``int32x2_t vadd_s32 (int32x2_t a, int32x2_t b)``
- ``int32x4_t vaddq_s32 (int32x4_t a, int32x4_t b)``

mul
---

- ``int32x4_t vmulq_s32 (int32x4_t a, int32x4_t b)``
- ``int32x2_t vmul_s32 (int32x2_t a, int32x2_t b)``
- ``int32x2_t vmul_n_s32 (int32x2_t a, int32_t b)``
- ``int32x4_t vmulq_n_s32 (int32x4_t a, int32_t b)``

- ``int32x2_t vmla_s32 (int32x2_t a, int32x2_t b, int32x2_t c)``

    return ``a + b * c``

- ``int32x4_t vmlaq_s32 (int32x4_t a, int32x4_t b, int32x4_t c)``

- ``float32x4_t vmlaq_f32 (float32x4_t a, float32x4_t b, float32x4_t c)``

  .. HINT::

    ``float32x4_t vfmaq_f32 (float32x4_t a, float32x4_t b, float32x4_t c)`` is preferred
    for floatting point multiply-and-accumulate.

- ``float32x2_t vmla_f32 (float32x2_t a, float32x2_t b, float32x2_t c)``

  .. HINT::

    ``float32x2_t vfma_f32 (float32x2_t a, float32x2_t b, float32x2_t c)`` is preferred
    for floatting point multiply-and-accumulate.

cvt
---

- ``int32x2_t  vcvt_s32_f32(float32x2_t a)``
- ``int32x4_t vcvtq_s32_f32 (float32x4_t a)``


References
----------

- `Summary of NEON intrinsics <http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491i/CIHJBEFE.html>`_

