new
===

Summary
-------

Pay attetion to ``new expression`` and ``operator new``.

- ``new int``, ``new int{}``, ``new int{2}``, ``new Foo`` invokes ``void* operator new(size_t);``
- ``p = new int; delete p;`` invokes ``void  operator delete(void* ptr, std::size_t size)``
- ``p = new Foo; delete p;`` invokes ``void  operator delete(void* ptr, std::size_t size)``
- ``new int[2]`` invokes ``void* operator new[](std::size_t size);``
- ``p = new int[2]; delete [] p;`` invokes ``void  operator delete[](void* ptr)``

For placement new (we have to include ``<new>``):

- ``int a; int* p = new(&a) int; `` invokes ``void* operator new  (std::size_t, void* __p)``

Each kind of ``operator new`` as the corresponding ``operator delete``.

The first argument of ``operator new`` is always ``std::size_t``, while the first argument of
``operator delete`` is ``void *``. 

There is only ``placement new`` but no ``placement delete``. There are only ``delete p``
and ``delete []p``.

There are ``placement operator new`` and ``placement operator delete``.



Example 1
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int;
      p[0] = 1;
  }

The compiler output is (see `<https://godbolt.org/>`_):

.. code-block:: c++

  Test ()
  {
    int * p;

    p = operator new (4);
    *p = 1;
  }


The assembly output is:

.. code-block:: gas

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $4, %edi
          call    operator new(unsigned long)
          movq    %rax, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $1, (%rax)
          nop
          leave
          ret

Note that it calls ``void* operator new(size_t);``
From ``libcxx/include/new``:


.. code-block:: c++

  void* operator new(std::size_t size);

From ``libcxx/src/new.cpp``:

.. code-block:: c++

  _LIBCPP_WEAK
  void *
  operator new(std::size_t size) _THROW_BAD_ALLOC
  {
      if (size == 0)
          size = 1;
      void* p;
      while ((p = ::malloc(size)) == 0)
      {
          // If malloc fails and there is a new_handler,
          // call it to try free up memory.
          std::new_handler nh = std::get_new_handler();
          if (nh)
              nh();
          else
  #ifndef _LIBCPP_NO_EXCEPTIONS
              throw std::bad_alloc();
  #else
              break;
  #endif
      }
      return p;
  }

We can see that:

  - (1) If size is 0, it is assigned to 1, which means ``operator new`` always allocates with a non-zero size.
  - (2) It uses ``malloc`` from libc to allocate memory
  - (3) If ``malloc`` returns a nullptr, it will call ``std::get_new_handler`` to handle OOM.
  - (4) It throws ``std::bad_alloc`` at the end.

``new_handler`` is defined as:

.. code-block:: c++

  typedef void (*new_handler)();

From ``libcxx/src/support/runtime/new_handler_fallback.ipp``:

.. code-block:: c++

  namespace std {

  _LIBCPP_SAFE_STATIC static std::new_handler __new_handler;

  new_handler
  set_new_handler(new_handler handler) _NOEXCEPT
  {
      return __libcpp_atomic_exchange(&__new_handler, handler);
  }

  new_handler
  get_new_handler() _NOEXCEPT
  {
      return __libcpp_atomic_load(&__new_handler);
  }

  } // namespace std

Example 2
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int;
      p[0] = 1;
      delete p;
  }

The output of the compiler is:

.. code-block:: c++

  Test ()
  {
    int * p.0;
    int * p;

    p = operator new (4);
    *p = 1;
    p.0 = p;
    if (p.0 != 0B) goto <D.2335>; else goto <D.2336>;
    <D.2335>:
    try
      {
        *p.0 = {CLOBBER};
      }
    finally
      {
        operator delete (p.0, 4);
      }
    goto <D.2337>;
    <D.2336>:
    <D.2337>:
  }

The assembly output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $4, %edi
          call    operator new(unsigned long)
          movq    %rax, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $1, (%rax)
          movq    -8(%rbp), %rax
          testq   %rax, %rax
          je      .L3
          movl    $4, %esi
          movq    %rax, %rdi
          call    operator delete(void*, unsigned long)
  .L3:
          nop
          leave
          ret

For ``delete p``:

  - (1) It tests whether ``p`` is 0 or not.
  - (2) If it is 0, then it is is no-op
  - (3) Otherwise, it uses a ``try ... finally`` statement
  - (4) It calls ``void  operator delete(void* ptr, std::size_t size)``
        in ``finally`` to delete the pointer.

From ``libcxx/src/new.cpp``:

.. code-block:: c++

  _LIBCPP_WEAK
  void
  operator delete(void* ptr, size_t) _NOEXCEPT
  {
      ::operator delete(ptr);
  }

  _LIBCPP_WEAK
  void
  operator delete(void* ptr) _NOEXCEPT
  {
      ::free(ptr);
  }

We can see that ``operator delete`` invokes ``free``.


Example 3
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int[3];
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    int * p;

    p = operator new [] (12);
  }

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $12, %edi
          call    operator new[](unsigned long)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

We can see that ``new int[3]`` expands to the following:

  - It calls ``void* operator new[](std::size_t size);``
  - ``size`` is 12.

From ``libcxx/src/new/cpp``,

.. code-block:: c++

  _LIBCPP_WEAK
  void*
  operator new[](size_t size) _THROW_BAD_ALLOC
  {
      return ::operator new(size);
  }

Note: ``operator new[]`` invokes ``operator new``

Example 4
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int[3];
      delete []p;
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    int * p;

    p = operator new [] (12);
    if (p != 0B) goto <D.2334>; else goto <D.2335>;
    <D.2334>:
    operator delete [] (p);
    goto <D.2336>;
    <D.2335>:
    <D.2336>:
  }

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $12, %edi
          call    operator new[](unsigned long)
          movq    %rax, -8(%rbp)
          cmpq    $0, -8(%rbp)
          je      .L3
          movq    -8(%rbp), %rax
          movq    %rax, %rdi
          call    operator delete[](void*)
  .L3:
          nop
          leave
          ret

Note: ``delete []p`` invokes: ``void  operator delete[](void* ptr)``.

From ``libcxx/src/new.cpp``:

.. code-block:: c++

  _LIBCPP_WEAK
  void
  operator delete[] (void* ptr) _NOEXCEPT
  {
      ::operator delete(ptr);
  }

Example 5
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int{10};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2334;
    int * p;

    D.2334 = operator new (4);
    try
      {
        MEM[(int *)D.2334] = 10;
      }
    catch
      {
        operator delete (D.2334, 4);
      }
    p = D.2334;
  }

Note: ``new int{10}`` is more complicated than ``new int``.

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $4, %edi
          call    operator new(unsigned long)
          movl    $10, (%rax)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

Example 6
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int{};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2334;
    int * p;

    D.2334 = operator new (4);
    try
      {
        MEM[(int *)D.2334] = 0;
      }
    catch
      {
        operator delete (D.2334, 4);
      }
    p = D.2334;
  }

You see that ``new int{}`` is equivalent to ``new int{0}``.

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $4, %edi
          call    operator new(unsigned long)
          movl    $0, (%rax)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

Example 7
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int[1]{};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2334;
    int * retval.0;
    int * D.2335;
    int * D.2336;
    long int D.2337;
    int * p;

    D.2334 = operator new [] (4);
    try
      {
        D.2335 = D.2334;
        D.2336 = D.2335;
        D.2337 = 0;
        <D.2341>:
        if (D.2337 < 0) goto <D.2338>; else goto <D.2342>;
        <D.2342>:
        *D.2336 = 0;
        D.2336 = D.2336 + 4;
        D.2337 = D.2337 + -1;
        goto <D.2341>;
        <D.2338>:
        retval.0 = D.2335;
      }
    catch
      {
        operator delete [] (D.2334);
      }
    p = D.2334;
  }

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $4, %edi
          call    operator new[](unsigned long)
          movq    %rax, %rcx
          movq    %rcx, %rdx
          movl    $0, %eax
  .L3:
          testq   %rax, %rax
          js      .L2
          movl    $0, (%rdx)
          addq    $4, %rdx
          subq    $1, %rax
          jmp     .L3
  .L2:
          movq    %rcx, -8(%rbp)
          nop
          leave
          ret

Example 8
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int[2]{};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2334;
    int * retval.0;
    int * D.2335;
    int * D.2336;
    long int D.2337;
    int * p;

    D.2334 = operator new [] (8);
    try
      {
        D.2335 = D.2334;
        D.2336 = D.2335;
        D.2337 = 1;
        <D.2341>:
        if (D.2337 < 0) goto <D.2338>; else goto <D.2342>;
        <D.2342>:
        *D.2336 = 0;
        D.2336 = D.2336 + 4;
        D.2337 = D.2337 + -1;
        goto <D.2341>;
        <D.2338>:
        retval.0 = D.2335;
      }
    catch
      {
        operator delete [] (D.2334);
      }
    p = D.2334;
  }

Note: The code for ``new int[1]{}`` is the same as ``new int[2]{}``.


Example 9
---------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int[2]{100};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2334;
    int * retval.0;
    int * D.2335;
    int * D.2336;
    long int D.2337;
    int * p;

    D.2334 = operator new [] (8);
    try
      {
        D.2335 = D.2334;
        D.2336 = D.2335;
        D.2337 = 1;
        *D.2336 = 100;
        D.2336 = D.2336 + 4;
        D.2337 = D.2337 + -1;
        <D.2341>:
        if (D.2337 < 0) goto <D.2338>; else goto <D.2342>;
        <D.2342>:
        *D.2336 = 0;
        D.2336 = D.2336 + 4;
        D.2337 = D.2337 + -1;
        goto <D.2341>;
        <D.2338>:
        retval.0 = D.2335;
      }
    catch
      {
        operator delete [] (D.2334);
      }
    p = D.2334;
  }

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $8, %edi
          call    operator new[](unsigned long)
          movq    %rax, %rcx
          movq    %rcx, %rax
          movl    $1, %esi
          movl    $100, (%rax)
          leaq    4(%rax), %rdx
          leaq    -1(%rsi), %rax
  .L3:
          testq   %rax, %rax
          js      .L2
          movl    $0, (%rdx)
          addq    $4, %rdx
          subq    $1, %rax
          jmp     .L3
  .L2:
          movq    %rcx, -8(%rbp)
          nop
          leave
          ret

Example 10
----------

For the following code:

.. code-block:: c++

  void Test() {
      int* p = new int[2]{100, 0};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2334;
    int * retval.0;
    int * D.2335;
    int * D.2336;
    long int D.2337;
    int * p;

    D.2334 = operator new [] (8);
    try
      {
        D.2335 = D.2334;
        D.2336 = D.2335;
        D.2337 = 1;
        *D.2336 = 100;
        D.2336 = D.2336 + 4;
        D.2337 = D.2337 + -1;
        *D.2336 = 0;
        D.2336 = D.2336 + 4;
        D.2337 = D.2337 + -1;
        retval.0 = D.2335;
      }
    catch
      {
        operator delete [] (D.2334);
      }
    p = D.2334;
  }

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $8, %edi
          call    operator new[](unsigned long)
          movq    %rax, %rdx
          movl    $100, (%rdx)
          addq    $4, %rdx
          movl    $0, (%rdx)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

Example 11
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
  };

  void Test() {
      Foo* p = new Foo;
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    struct Foo * p;

    p = operator new (12);
  }


Note: It does not invoke the constructor of ``Foo``.

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $12, %edi
          call    operator new(unsigned long)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

Example 12
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
  };

  void Test() {
      Foo* p = new Foo{};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2359;
    struct Foo * p;

    D.2359 = operator new (12);
    try
      {
        MEM[(struct Foo *)D.2359] = {};
      }
    catch
      {
        operator delete (D.2359, 12);
      }
    p = D.2359;
  }

Note: ``new Foo{}`` will zero initialize ``Foo``.

Example 13
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
  };

  void Test() {
      Foo* p = new Foo{1,2,3};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2359;
    struct Foo * p;

    D.2359 = operator new (12);
    try
      {
        MEM[(struct Foo *)D.2359].i = 1;
        MEM[(struct Foo *)D.2359].a = 2;
        MEM[(struct Foo *)D.2359].k = 3;
      }
    catch
      {
        operator delete (D.2359, 12);
      }
    p = D.2359;
  }

Note: It still does not invoke the constructor of ``Foo``.

The assembler output is:

.. code-block::

  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          movl    $12, %edi
          call    operator new(unsigned long)
          movl    $1, (%rax)
          movb    $2, 4(%rax)
          movl    $3, 8(%rax)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

Example 14
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
      Foo() {i = 1; a = 2; k = 3;}
  };

  void Test() {
      Foo* p = new Foo;
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2347;
    struct Foo * p;

    D.2347 = operator new (12);
    try
      {
        Foo::Foo (D.2347);
      }
    catch
      {
        operator delete (D.2347, 12);
      }
    p = D.2347;
  }


  Foo::Foo (struct Foo * const this)
  {
    *this = {CLOBBER};
    {
      this->i = 1;
      this->a = 2;
      this->k = 3;
    }
  }

Note: Now it calls the constructor of ``Foo``.

The assembler output is:

.. code-block::

  Foo::Foo() [base object constructor]:
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $1, (%rax)
          movq    -8(%rbp), %rax
          movb    $2, 4(%rax)
          movq    -8(%rbp), %rax
          movl    $3, 8(%rax)
          nop
          popq    %rbp
          ret
  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          pushq   %rbx
          subq    $24, %rsp
          movl    $12, %edi
          call    operator new(unsigned long)
          movq    %rax, %rbx
          movq    %rbx, %rdi
          call    Foo::Foo() [complete object constructor]
          movq    %rbx, -24(%rbp)
          nop
          movq    -8(%rbp), %rbx
          leave
          ret

Example 15
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
      Foo() {i = 1; a = 2; k = 3;}
  };

  void Test() {
      Foo* p = new Foo[2];
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2347;
    struct Foo * retval.0;
    struct Foo * D.2348;
    struct Foo * D.2349;
    long int D.2350;
    struct Foo * p;

    D.2347 = operator new [] (24);
    try
      {
        D.2348 = D.2347;
        D.2349 = D.2348;
        D.2350 = 1;
        <D.2372>:
        if (D.2350 < 0) goto <D.2369>; else goto <D.2373>;
        <D.2373>:
        Foo::Foo (D.2349);
        D.2349 = D.2349 + 12;
        D.2350 = D.2350 + -1;
        goto <D.2372>;
        <D.2369>:
        retval.0 = D.2348;
      }
    catch
      {
        operator delete [] (D.2347);
      }
    p = D.2347;
  }


  Foo::Foo (struct Foo * const this)
  {
    *this = {CLOBBER};
    {
      this->i = 1;
      this->a = 2;
      this->k = 3;
    }
  }

The assembler output is:

.. code-block::

  Foo::Foo() [base object constructor]:
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $1, (%rax)
          movq    -8(%rbp), %rax
          movb    $2, 4(%rax)
          movq    -8(%rbp), %rax
          movl    $3, 8(%rax)
          nop
          popq    %rbp
          ret
  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          pushq   %r13
          pushq   %r12
          pushq   %rbx
          subq    $24, %rsp
          movl    $24, %edi
          call    operator new[](unsigned long)
          movq    %rax, %r13
          movq    %r13, %r12
          movl    $1, %ebx
  .L4:
          testq   %rbx, %rbx
          js      .L3
          movq    %r12, %rdi
          call    Foo::Foo() [complete object constructor]
          addq    $12, %r12
          subq    $1, %rbx
          jmp     .L4
  .L3:
          movq    %r13, -40(%rbp)
          nop
          addq    $24, %rsp
          popq    %rbx
          popq    %r12
          popq    %r13
          popq    %rbp
          ret

Example 16
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
      Foo() {i = 1; a = 2; k = 3;}
  };

  void Test() {
      Foo* p = new Foo;
      delete p;
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2347;
    struct Foo * p.0;
    struct Foo * p;

    D.2347 = operator new (12);
    try
      {
        Foo::Foo (D.2347);
      }
    catch
      {
        operator delete (D.2347, 12);
      }
    p = D.2347;
    p.0 = p;
    if (p.0 != 0B) goto <D.2376>; else goto <D.2377>;
    <D.2376>:
    try
      {
        *p.0 = {CLOBBER};
      }
    finally
      {
        operator delete (p.0, 12);
      }
    goto <D.2378>;
    <D.2377>:
    <D.2378>:
  }

  Foo::Foo (struct Foo * const this)
  {
    *this = {CLOBBER};
    {
      this->i = 1;
      this->a = 2;
      this->k = 3;
    }
  }

Note: It does not call the destructor of ``Foo``.

The assembler output is:

.. code-block::

  Foo::Foo() [base object constructor]:
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $1, (%rax)
          movq    -8(%rbp), %rax
          movb    $2, 4(%rax)
          movq    -8(%rbp), %rax
          movl    $3, 8(%rax)
          nop
          popq    %rbp
          ret
  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          pushq   %rbx
          subq    $24, %rsp
          movl    $12, %edi
          call    operator new(unsigned long)
          movq    %rax, %rbx
          movq    %rbx, %rdi
          call    Foo::Foo() [complete object constructor]
          movq    %rbx, -24(%rbp)
          movq    -24(%rbp), %rax
          testq   %rax, %rax
          je      .L4
          movl    $12, %esi
          movq    %rax, %rdi
          call    operator delete(void*, unsigned long)
  .L4:
          nop
          movq    -8(%rbp), %rbx
          leave
          ret

Example 16
----------

For the following code:

.. code-block:: c++

  struct Foo {
      int i;
      char a;
      int k;
      Foo() {i = 1; a = 2; k = 3;}
      ~Foo() {i = 100; a= 200;}
  };

  void Test() {
      Foo* p = new Foo;
      delete p;
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.2363;
    struct Foo * p.0;
    struct Foo * p;

    D.2363 = operator new (12);
    try
      {
        Foo::Foo (D.2363);
      }
    catch
      {
        operator delete (D.2363, 12);
      }
    p = D.2363;
    p.0 = p;
    if (p.0 != 0B) goto <D.2374>; else goto <D.2375>;
    <D.2374>:
    try
      {
        Foo::~Foo (p.0);
      }
    finally
      {
        operator delete (p.0, 12);
      }
    goto <D.2376>;
    <D.2375>:
    <D.2376>:
  }


  Foo::Foo (struct Foo * const this)
  {
    *this = {CLOBBER};
    {
      this->i = 1;
      this->a = 2;
      this->k = 3;
    }
  }


  Foo::~Foo (struct Foo * const this)
  {
    try
      {
        {
          try
            {
              this->i = 100;
              this->a = -56;
            }
          finally
            {
              *this = {CLOBBER};
            }
        }
        <D.2357>:
      }
    catch
      {
        <<<eh_must_not_throw (terminate)>>>
      }
  }

Note: Now it invokes the destructor of ``Foo``.

The assembler output is:

.. code-block::

  Foo::Foo() [base object constructor]:
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $1, (%rax)
          movq    -8(%rbp), %rax
          movb    $2, 4(%rax)
          movq    -8(%rbp), %rax
          movl    $3, 8(%rax)
          nop
          popq    %rbp
          ret
  Foo::~Foo() [base object destructor]:
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    -8(%rbp), %rax
          movl    $100, (%rax)
          movq    -8(%rbp), %rax
          movb    $-56, 4(%rax)
          nop
          popq    %rbp
          ret
  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          pushq   %rbx
          subq    $24, %rsp
          movl    $12, %edi
          call    operator new(unsigned long)
          movq    %rax, %rbx
          movq    %rbx, %rdi
          call    Foo::Foo() [complete object constructor]
          movq    %rbx, -24(%rbp)
          movq    -24(%rbp), %rbx
          testq   %rbx, %rbx
          je      .L5
          movq    %rbx, %rdi
          call    Foo::~Foo() [complete object destructor]
          movl    $12, %esi
          movq    %rbx, %rdi
          call    operator delete(void*, unsigned long)
  .L5:
          nop
          movq    -8(%rbp), %rbx
          leave
          ret

Example 17
----------

For the following code:

.. code-block:: c++

  #include<new>
  void Test() {
      int a;
      int* p = new (&a) int;
  }

Note: We have to include ``<new>`` to use placement new.

The compiler output is:

.. code-block:: c++

  Test ()
  {
    int * D.6552;
    int a;
    int * p;

    try
      {
        D.6552 = &a;
        p = operator new (4, D.6552);
      }
    finally
      {
        a = {CLOBBER};
      }
  }


  operator new (size_t D.6514, void * __p)
  {
    void * D.6553;

    try
      {
        D.6553 = __p;
        return D.6553;
      }
    catch
      {
        <<<eh_must_not_throw (terminate)>>>
      }
  }

Note: It uses placement new. From ``libcxx/include/new``:

.. code-block:: c++

  _LIBCPP_NODISCARD_AFTER_CXX17 inline _LIBCPP_INLINE_VISIBILITY void* operator new  (std::size_t, void* __p) _NOEXCEPT {return __p;}

The assembler output is:

.. code-block::

  operator new(unsigned long, void*):
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    %rsi, -16(%rbp)
          movq    -16(%rbp), %rax
          popq    %rbp
          ret
  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          leaq    -12(%rbp), %rax
          movq    %rax, %rsi
          movl    $4, %edi
          call    operator new(unsigned long, void*)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

Example 18
----------

For the following code:

.. code-block:: c++

  #include<new>

  void Test() {
      int a;
      int*p = new (&a) int{3};
  }

The compiler output is:

.. code-block:: c++

  Test ()
  {
    void * D.6554;
    void * D.6553;
    int a;
    int * p;

    try
      {
        D.6554 = &a;
        D.6553 = operator new (4, D.6554);
        try
          {
            MEM[(int *)D.6553] = 3;
          }
        catch
          {
            operator delete (D.6553, D.6554);
          }
        p = D.6553;
      }
    finally
      {
        a = {CLOBBER};
      }
  }


  operator new (size_t D.6514, void * __p)
  {
    void * D.6555;

    try
      {
        D.6555 = __p;
        return D.6555;
      }
    catch
      {
        <<<eh_must_not_throw (terminate)>>>
      }
  }

Note: It calls ``placement operator delete`` inside ``catch``:

.. code-block:: c++

  void  operator delete  (void*, void*)

From ``libcxx/inclue/new``,

.. code-block:: c++

  inline _LIBCPP_INLINE_VISIBILITY void  operator delete  (void*, void*) _NOEXCEPT {}

The assembler output is:

.. code-block::

  operator new(unsigned long, void*):
          pushq   %rbp
          movq    %rsp, %rbp
          movq    %rdi, -8(%rbp)
          movq    %rsi, -16(%rbp)
          movq    -16(%rbp), %rax
          popq    %rbp
          ret
  Test():
          pushq   %rbp
          movq    %rsp, %rbp
          subq    $16, %rsp
          leaq    -12(%rbp), %rax
          movq    %rax, %rsi
          movl    $4, %edi
          call    operator new(unsigned long, void*)
          movl    $3, (%rax)
          movq    %rax, -8(%rbp)
          nop
          leave
          ret

