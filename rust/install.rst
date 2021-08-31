Installation
============

See `<https://doc.rust-lang.org/book/ch01-01-installation.html>`_

macOS
-----

.. code-block::

  curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

It installs the following files inside ``$HOME/.cargo/bin/rustc``:

  - bin/cargo
  - bin/rustc
  - bin/rust-gdb
  - bin/rustup
  - bin/rustfmt
  - ...

To update the installation, run ``rustup update``. To print the version
of rust, run ``rustc --version``, which prints something like::

  rustc 1.53.0 (53cb7b09b 2021-06-17)

Rust users are known as Rustaceans.

Run ``rustup doc`` view the documentation **offline** in a browser.
