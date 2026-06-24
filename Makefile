# -- Configuration --
# Easily expand this by adding space-separated directories
DOCS := ghghnd papers

# -- Phony Targets --
.PHONY: all pdf html clean help

# Create dynamic target lists: all-book, all-papers, pdf-book, etc.
ALL_TARGETS   := $(addprefix all-,$(DOCS))
PDF_TARGETS   := $(addprefix pdf-,$(DOCS))
HTML_TARGETS  := $(addprefix html-,$(DOCS))
CLEAN_TARGETS := $(addprefix clean-,$(DOCS))

# Hook the main commands up to their sub-target groups
all: $(ALL_TARGETS)
pdf: $(PDF_TARGETS)
html: $(HTML_TARGETS)
clean: $(CLEAN_TARGETS)

# Rules to execute inside each directory
$(ALL_TARGETS): all-%:
	$(MAKE) -C $* all

$(PDF_TARGETS): pdf-%:
	$(MAKE) -C $* pdf

$(HTML_TARGETS): html-%:
	$(MAKE) -C $* html

$(CLEAN_TARGETS): clean-%:
	$(MAKE) -C $* clean

# Helper for common commands
help:
	@echo "The Abstract Universe Project - Build System"
	@echo "-------------------------------------------"
	@echo "make all       - Build both PDFs and HTML"
	@echo "make pdf       - Build PDFs only"
	@echo "make html      - Build HTML only"
	@echo "make clean     - Remove build artifacts"

