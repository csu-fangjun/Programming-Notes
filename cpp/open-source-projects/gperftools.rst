
gperftools
==========


GitHub repository: `<https://github.com/gperftools/gperftools>`_

.. code-block::

  wget http://download.savannah.gnu.org/releases/libunwind/libunwind-0.99-beta.tar.gz
  tar xf libunwind-0.99-beta.tar.gz
  cd libunwind-0.99-beta

  # to fix error: 'longjmp' aliased to undefined symbol '_longjmp'
  # add CFLAGS="-m64"

  CFLAGS="-m64" ./configure --prefix=/path/to/libunwind
  make install

  git clone --depth 1 https://github.com/gperftools/gperftools.git
  cd gperftools
  ./autogen.sh

  export CPPFLAGS=-I/path/to/installed/libunwind/include
  export LDFLAGS=-L/path/to/installed/libunwind/lib
  ./configure --prefix=/path/to/gperftools

pprof
-----

GitHub repository: `<https://github.com/google/pprof>`_

Usage:

.. code-block:: bash

  pprof -http ip:port -no_browser exe_binary cpu_prof_filename

  pprof -pdf exe_binary cpu_prof_filename
