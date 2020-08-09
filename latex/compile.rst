
Compile
=======

**latexmk**
  - ``latexmk --help``

  .. code-block::

    .PHONY: all clean
    all: main.tex
      latexmk -xelatex main.tex

    clean:
      latexmk -C # it cleans also pdfs
      latexmk -c

**div**
  - ``dvipdfm story.dvi``, it produces ``story.pdf``.
  - ``latex story.tex``, it produces ``story.dvi``
  - ``pdflatex story.tex``, it produces ``story.pdf``
  - ``dvips story.dvi``, it produces ``story.ps``
  - ``ps2pdf story.ps``, it produces ``story.pdf``
