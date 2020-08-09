
Protobuf
========

- Download Protocol Buffers

    `<https://developers.google.com/protocol-buffers/docs/downloads>`_

.. code-block::

  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.12.1/protobuf-all-3.12.1.tar.gz
  tar xf protobuf-all-3.12.1.tar.gz
  cd protobuf-all-3.12.1

Read ``README.md`` and ``src/README.md`` for installation.

The protobuf release has the following format: ``protobuf-xxx-version.tar.gz``,
where ``xxx`` may be ``all``, ``cpp``, ``python``, ``java``.
If it is ``cpp``, then the ``.tar.gz`` contains only ``cpp``. If it is ``python``,
then it contains both ``cpp`` and ``python``. That is, ``cpp`` is always included.

.. code-block::

  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.12.1/protobuf-cpp-3.12.1.tar.gz
  tar xf protobuf-cpp-3.12.1.tar.gz
  cd protobuf-3.12.1

.. code-block::

  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.12.1/protobuf-python-3.12.1.tar.gz
  tar xf protobuf-python-3.12.1.tar.gz
  cd protobuf-3.12.1
  ./configure --help
  ./configure --prefix=$HOME/software/protobuf/3.12.1
  make -j2
  make install

``ls $HOME/software/protobuf/3.12.1/*`` prints:

.. code-block::

    /xxx/software/protobuf/3.12.1/bin:
    protoc

    /xxx/software/protobuf/3.12.1/include:
    google

    /xxx/software/protobuf/3.12.1/lib:
    libprotobuf.a
    libprotobuf.la
    libprotobuf-lite.a
    libprotobuf-lite.la
    libprotobuf-lite.so
    libprotobuf-lite.so.23
    libprotobuf-lite.so.23.0.1
    libprotobuf.so
    libprotobuf.so.23
    libprotobuf.so.23.0.1
    libprotoc.a
    libprotoc.la
    libprotoc.so
    libprotoc.so.23
    libprotoc.so.23.0.1
    pkgconfig

``ls include/google/protobuf`` displays:

.. code-block::

  any.h                  descriptor.proto             generated_message_reflection.h    map.h               reflection_ops.h      type.proto
  any.pb.h               duration.pb.h                generated_message_table_driven.h  map_type_handler.h  repeated_field.h      unknown_field_set.h
  any.proto              duration.proto               generated_message_util.h          me                  service.h             util
  api.pb.h               dynamic_message.h            has_bits.h                        message.h           source_context.pb.h   wire_format.h
  api.proto              empty.pb.h                   implicit_weak_message.h           message_lite.h      source_context.proto  wire_format_lite.h
  arena.h                empty.proto                  inlined_string_field.h            metadata.h          struct.pb.h           wrappers.pb.h
  arena_impl.h           extension_set.h              io                                metadata_lite.h     struct.proto          wrappers.proto
  arenastring.h          extension_set_inl.h          map_entry.h                       parse_context.h     stubs
  compiler               field_mask.pb.h              map_entry_lite.h                  port_def.inc        text_format.h
  descriptor_database.h  field_mask.proto             map_field.h                       port.h              timestamp.pb.h
  descriptor.h           generated_enum_reflection.h  map_field_inl.h                   port_undef.inc      timestamp.proto
  descriptor.pb.h        generated_enum_util.h        map_field_lite.h                  reflection.h        type.pb.h


Note that ``protoc`` is inside the ``bin`` directory and we have to add it to ``PATH``:

.. code-block::

  export PATH=$HOME/software/protobuf/3.12.1/bin:${PATH}

pkg-config
----------

.. code-block::

  software/protobuf/3.12.1/lib/pkgconfig$ ls

  protobuf-lite.pc  protobuf.pc

.. code-block::

    ~/software/protobuf/3.12.1/lib/pkgconfig$ cat protobuf-lite.pc
    prefix=/xxx/software/protobuf/3.12.1
    exec_prefix=${prefix}
    libdir=${exec_prefix}/lib
    includedir=${prefix}/include

    Name: Protocol Buffers
    Description: Google's Data Interchange Format
    Version: 3.12.1
    Libs: -L${libdir} -lprotobuf-lite
    Cflags: -I${includedir} -pthread
    Conflicts: protobuf

    ~/software/protobuf/3.12.1/lib/pkgconfig$ cat protobuf.pc
    prefix=/xxx/software/protobuf/3.12.1
    exec_prefix=${prefix}
    libdir=${exec_prefix}/lib
    includedir=${prefix}/include

    Name: Protocol Buffers
    Description: Google's Data Interchange Format
    Version: 3.12.1
    Libs: -L${libdir} -lprotobuf
    Libs.private: -lz

    Cflags: -I${includedir} -pthread
    Conflicts: protobuf-lite

To use it in ``Makefile``, we have to first change ``PKG_CONFIG_PATH``

.. code-block::

    export PKG_CONFIG_PATH=$HOME/software/protobuf/3.12.1/lib/pkgconfig:${PKG_CONFIG_PATH}

Then the following commands should print:

.. code-block:: console

  $ pkg-config --cflags protobuf
  -pthread -I/xxx/software/protobuf/3.12.1/include

  $ pkg-config --libs protobuf
  -L/xxx/software/protobuf/3.12.1/lib -lprotobuf

  $ pkg-config --cflags-only-I protobuf
  -I/xxx/software/protobuf/3.12.1/include

  $ pkg-config --cflags-only-other protobuf
  -pthread

  $ pkg-config --libs-only-l protobuf
  -lprotobuf

  $ pkg-config --libs-only-L protobuf
  -L/xxx/software/protobuf/3.12.1/lib

  $ pkg-config --libs-only-other protobuf
  <empty output>

Examples
--------

Refer to ``Example/Makefile`` for how to compile an exmaple.

protoc
------

.. code-block::

  $ protoc --help
  $ mkdir ./abc
  $ protoc addressbook.proto --cpp_out=./abc

It creates ``./abc/addressbook.pb.cc`` and ``./abc/addressbook.pb.h``.


.. note::

  We have to link our program with ``-pthread`` while using ``libprotobuf``;
  otherwise, the following runtime error will occur:

  .. code-block::

      [libprotobuf FATAL google/protobuf/generated_message_util.cc:812] CHECK failed: (scc->visit_status.load(std::memory_order_relaxed)) == (SCCInfoBase::kRunning):
      terminate called after throwing an instance of 'google::protobuf::FatalException'
        what():  CHECK failed: (scc->visit_status.load(std::memory_order_relaxed)) == (SCCInfoBase::kRunning):
        Aborted (core dumped)

.. literalinclude:: ./code/protobuf/Makefile
  :caption: Makefile
  :language: makefile
  :linenos:

.. code-block:: console

    $ make
    protoc addressbook.proto --cpp_out=./abc
    g++ -c -pthread -I/xxx/software/protobuf/3.12.1/include -std=c++11 -g hello.cc -o hello.o
    g++ -c -pthread -I/xxx/software/protobuf/3.12.1/include -std=c++11 -g abc/addressbook.pb.cc -o abc/addressbook.pb.o
    g++ -o hello hello.o abc/addressbook.pb.o -L/xxx/software/protobuf/3.12.1/lib -lprotobuf "-Wl,-rpath,/home/fangjunkuang/software/protobuf/3.12.1/lib" -pthread
