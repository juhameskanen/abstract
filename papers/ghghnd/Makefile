DOCS = ghghnd iame

PDFS      = $(DOCS:=.pdf)
HTML_DIR  = $(addprefix html-output-, $(DOCS))
REMOTE_DIR = /meskanen/abstract


ghghnd.pdf : PROF_MACRO = \def\buildtheology{true}
iame.pdf   : PROF_MACRO = \def\buildtheology{false}


html-output-ghghnd : PROF_MACRO = \def\buildtheology{true}
html-output-iame   : PROF_MACRO = \def\buildtheology{false}


all: $(PDFS) $(HTML_DIR)
	@echo "All documents successfully compiled!"

$(DOCS): %: %.pdf html-output-%


%.pdf: %.tex
	@echo "Building $@ with profile flags..."
	latexmk -pdf -quiet -interaction=nonstopmode \
		-pdflatex="pdflatex %O '$(PROF_MACRO)\input{%S}'" \
		$<

html-output-%: %.tex
	@echo "Building HTML profile $*..."


clean:
	@for doc in $(DOCS); do \
		latexmk -C -jobname=$$doc $$doc.tex 2>/dev/null; \
		rm -rf html-output-$$doc; \
	done

.PHONY: all clean $(DOCS) upload

# separate file for security 
include upload.mak

