# Common makefile for building papers

PDFS       := $(DOCS:=.pdf)
REMOTE_DIR := /meskanen/abstract

TRASH_EXTS := 4ct 4tc css idv lg tmp xref dvi html svg bbl fdb_latexmk \
              aux glo ist fls toc run.xml bfc blg bcf log out

# --------------------------------------------------------
# Conditional HTML Target Mapping
# --------------------------------------------------------
ifdef NO_HTML
    # If NO_HTML is defined by the caller, only build the PDFs
    ALL_TARGETS := $(PDFS)
    HTML_DIRS   :=
else
    # Otherwise, default to building everything
    HTML_DIRS   := $(addprefix html-, $(DOCS))
    ALL_TARGETS := $(PDFS) $(HTML_DIRS)
endif
# --------------------------------------------------------

.PHONY: all clean purge $(DOCS) upload

all: $(ALL_TARGETS)
	@echo "All documents successfully compiled!"

%.pdf: %.tex
	@echo "Building $@ with profile flags..."
	latexmk -pdf -quiet -interaction=nonstopmode \
	    -pdflatex="pdflatex %O '$(PROF_MACRO)\input{%S}'" $<

# This target pattern is safe; if $(HTML_DIRS) is empty, this rule simply won't trigger
$(HTML_DIRS): html-%: %.tex
	@echo "Building $@..."
	make4ht -c $*.cfg -d $@ $< "xhtml,2,sections+,next"
	rm -f $(addprefix $*., $(TRASH_EXTS)) $*[0-9]*.* *ch[0-9]*.*

purge:
	@echo "Vaporizing temporary compilation trash..."
	rm -f $(foreach doc,$(DOCS),$(addprefix $(doc).,$(TRASH_EXTS)))
	rm -f $(foreach doc,$(DOCS),$(addprefix $(doc),*[0-9]*.* *ch[0-9]*.*))

clean: purge
	@echo "Removing valuable generated targets (PDFS and HTML folders)..."
	@for doc in $(DOCS); do \
	    latexmk -C $$doc.tex 2>/dev/null; \
	done
	rm -f $(PDFS)
	rm -rf $(addprefix html-, $(DOCS)) # Always wipe the directory on clean regardless