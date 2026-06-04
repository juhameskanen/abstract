DOCS       := ghghnd iame
PDFS       := $(DOCS:=.pdf)
HTML_DIRS  := $(addprefix html-, $(DOCS))
REMOTE_DIR := /meskanen/abstract

TRASH_EXTS := 4ct 4tc css idv lg tmp xref dvi html svg bbl fdb_latexmk \
              aux glo ist fls toc run.xml bfc blg bcf log out

ghghnd.pdf html-ghghnd : PROF_MACRO := \def\buildtheology{true}
iame.pdf   html-iame   : PROF_MACRO := \def\buildtheology{false}

.PHONY: all clean purge $(DOCS) upload

all: $(PDFS) $(HTML_DIRS)
	@echo "All documents successfully compiled!"

%.pdf: %.tex
	@echo "Building $@ with profile flags..."
	latexmk -pdf -quiet -interaction=nonstopmode \
	    -pdflatex="pdflatex %O '$(PROF_MACRO)\input{%S}'" $<

$(HTML_DIRS): html-%: %.tex
	@echo "Building $@..."
	make4ht -c $*.cfg -d $@ $< "xhtml,2,sections+,next"
	rm -f $(addprefix $*., $(TRASH_EXTS)) $*[0-9]*.* *ch[0-9]*.*

purge:
	@echo "Vaporizing temporary compilation trash..."
	@# Standard base names (e.g., ghghnd.4ct, ghghnd.bcf)
	rm -f $(foreach doc,$(DOCS),$(addprefix $(doc).,$(TRASH_EXTS)))
	@# Wildcard variations for the indexed/numbered trash (e.g., iame8x.svg, iamech7.html)
	rm -f $(foreach doc,$(DOCS),$(addprefix $(doc),*[0-9]*.* *ch[0-9]*.*))

clean: purge
	@echo "Removing valuable generated targets (PDFS and HTML folders)..."
	@for doc in $(DOCS); do \
		latexmk -C $$doc.tex 2>/dev/null; \
	done
	rm -f $(PDFS)
	rm -rf $(HTML_DIRS)


-include upload.mak
