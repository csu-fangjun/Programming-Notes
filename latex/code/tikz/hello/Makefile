

.PHONY: clean all

all:
	latexmk hello.tex  # generate hello.dvi
	dvisvgm --no-fonts hello.dvi hello.svg  # geneate hello.svg
	latexmk -pdf hello.tex # generate hello.pdf
	convert hello.pdf hello_convert.svg  # not recommended
	convert -density 300 hello.pdf -quality 90 hello.png
	convert -density 300 hello.pdf -quality 90 hello.jpg
	convert -density 300 hello.pdf -quality 90 -resize 50% -background white -alpha remove hello_50.jpg

clean:
	# $(RM) -v *.aux *.dvi *.fdb_latexmk *.fls *.log *.out
	latexmk -C # it cleans everything
	# latexmk -c # it does not clean *.pdf
	# $(RM) *.png *.jpg *.svg *.pdf


