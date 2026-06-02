DOCS = ghghnd iame

PDFS      = $(DOCS:=.pdf)
HTML_DIR  = $(addprefix html-, $(DOCS))
REMOTE_DIR = /meskanen/abstract


ghghnd.pdf : PROF_MACRO = \def\buildtheology{true}
iame.pdf   : PROF_MACRO = \def\buildtheology{false}


html-ghghnd : PROF_MACRO = \def\buildtheology{true}
html-iame   : PROF_MACRO = \def\buildtheology{false}


all: $(PDFS) $(HTML_DIR)
	@echo "All documents successfully compiled!"

$(DOCS): %: %.pdf html-%


%.pdf: %.tex
	@echo "Building $@ with profile flags..."
	latexmk -pdf -quiet -interaction=nonstopmode \
		-pdflatex="pdflatex %O '$(PROF_MACRO)\input{%S}'" \
		$<

$(HTML_DIR): html-%: %.tex
	@echo "Building html-$*..."
	make4ht -c $*.cfg -d html-$* $*.tex "xhtml,2,sections+,next"


clean:
	@echo "Cleaning up build artifacts..."
	@for doc in $(DOCS); do \
		latexmk -C $$doc.tex 2>/dev/null; \
	done
	make purge

purge:
	rm -f *.4ct *.4tc *.css *.idv *.lg *.tmp *.xref *.dvi *.html *.svg *.bbl *.fdb_latexmk *.aux *.glo *.ist *.fls *.toc *.run.xml *.bfc *.blg *.bcf

.PHONY: all clean $(DOCS) upload

# separate file for security 
include upload.mak

