tikz
====

Install tools for pdf
---------------------

.. code-block::

    sudo apt-get install imagemagick

It installs ``convert`` and other tools.

Edit::

    sudo vim /etc/ImageMagick-6/policy.xml

to solve the following error::

    $ convert -density 300 hello.pdf -quality 90 hello.png
    convert-im6.q16: not authorized `hello.pdf' @ error/constitute.c/ReadImage/412.
    convert-im6.q16: no images defined `hello.png' @ error/convert.c/ConvertImageCommand/3258.

replace::

    <policy domain="coder" rights="none" pattern="PDF" />

with::

    <policy domain="coder" rights="read|write" pattern="PDF" />


mogrify
-------

.. code-block::

    mogrify -density 300 -quality 98 -format png *.svg

It converts all ``*.svg`` files to ``*.png`` files.

See `<https://imagemagick.org/script/mogrify.php>`_ for its manual.

latexmk
--------

.. code-block::

    latexmk -C  # cleans every thing
    latexmk -c  # will not remove pdf, dvi

    latexmk hello.tex  # it generates hello.dvi
    dvips hello.dvi    # it generates hello.ps from hello.dvi
    ps2pdf hello.ps    # it generates hello.pdf from hello.ps

    latexmk -pdf hello.tex  # it generates hello.pdf

    # convert hello.pdf to hello.png
    # density specifies the dpi
    convert -density 300 hello.pdf -quality 90 hello.png

    convert -density 300 hello.pdf -quality 90 hello.jpg

    convert -density 300 hello.pdf -quality 90 -resize 50% -background white -alpha remove hello.jpg

    # convert hello.pdf to hello.svg
    convert hello.pdf hello.svg

xelatex
-------

.. code-block::

    xelatex hello.tex   # it generates hello.pdf

dvisvgm
-------

.. code-block::

    dvisvgm --no-fonts --page=1- multi.dvi
    # it generates multi-1.svg, multi-2.svg

.. code-block::

    dvisvgm --no-fonts --page=1- multi.dvi --output=test-%p.svg
    # It generates test-1.svg, test-2.svg

See `<https://man.archlinux.org/man/extra/texlive-bin/dvisvgm.1.en>`_
for available options.

References
----------

- Manual

  See `<https://github.com/pgf-tikz/pgf>`_

  The latest one can be found at `<https://pgf-tikz.github.io/pgf/pgfmanual.pdf>`_
