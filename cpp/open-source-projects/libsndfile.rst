
libsndfile
==========

.. code-block::

    sudo apt-get install libflac-dev libogg-dev
    sudo apt-get install libvorbis-dev
    wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz
    tar xf libsndfile-1.0.28.tar.gz
    cd libsndfile-1.0.28
    ./configure --prefix=/home/fangjunkuang/software/libsndfile
    make -j20 install

    export PKG_CONFIG_PATH=/home/fangjunkuang/software/libsndfile/lib/pkgcnfig:${PKG_CONFIG_PATH}


References for CMake: `<https://cmake.org/cmake/help/latest/module/FindPkgConfig.html>`_.

.. code-block::

    cmake_minimum_required(VERSION 3.11)

    find_package(PkgConfig REQUIRED)
    message(STATUS "PKG_CONFIG_PATH: $ENV{PKG_CONFIG_PATH}")
    pkg_check_modules(libsndfile REQUIRED sndfile)
    message(STATUS ${libsndfile_INCLUDE_DIRS})
    message(STATUS ${libsndfile_LIBRARY_DIRS})
    message(STATUS ${libsndfile_LDFLAGS})
    message(STATUS ${libsndfile_LINK_LIBRARIES})
    message(STATUS ${libsndfile_LIBRARIES})

Documentation for libsndfile:

- `<http://www.mega-nerd.com/libsndfile/api.html>`_
- `<https://github.com/erikd/libsndfile/issues/258>`_ for mp3.
