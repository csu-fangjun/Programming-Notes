.PHONY: clean all

all: multi.tex
	latexmk multi.tex  # generate multi.dvi
	dvisvgm --no-fonts --page=1- multi.dvi  # generates multi-1.svg, multi-2.svg
	mogrify -density 300 -quality 98 -format png *.svg

clean:
	# $(RM) -v *.aux *.dvi *.fdb_latexmk *.fls *.log *.out
	latexmk -C # it cleans everything
	# latexmk -c # it does not clean *.pdf
	$(RM) *.png *.jpg *.svg *.pdf

