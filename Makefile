# -- Configuration --
SCRIPTS_DIR := scripts
PDF_OUT     := pdf-output
HTML_OUT    := html
PAPERS_JSON := papers/papers.json

# -- Phony Targets --
.PHONY: all pdf html clean distclean help

# Default target: build everything
all: pdf html

# 1. Build PDFs
# This calls your specialized script which handles the jq-flattening and latexmk
pdf:
	@echo "==> Building PDFs via $(SCRIPTS_DIR)/build-pdfs.sh..."
	@chmod +x $(SCRIPTS_DIR)/build-pdf.sh
	@./$(SCRIPTS_DIR)/build-pdf.sh

# 2. Build HTML
# Depends on 'pdf' so that the HTML script finds the files in $(PDF_OUT)
html: pdf
	@echo "==> Building HTML via $(SCRIPTS_DIR)/build-pages.sh..."
	@chmod +x $(SCRIPTS_DIR)/build-pages.sh
	@./$(SCRIPTS_DIR)/build-pages.sh

# -- Cleanup --

# Removes LaTeX build artifacts from the source folders
clean:
	@echo "Cleaning up LaTeX build artifacts..."
	@find papers -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.bcf" \
		-o -name "*.blg" -o -name "*.log" -o -name "*.out" \
		-o -name "*.run.xml" -o -name "*.toc" -o -name "*.fdb_latexmk" \
		-o -name "*.fls" \) -delete

# Removes the generated output directories entirely
distclean: clean
	@echo "Removing generated folders ($(PDF_OUT) and $(HTML_OUT))..."
	@rm -rf $(PDF_OUT) $(HTML_OUT)

# Helper for common commands
help:
	@echo "The Abstract Universe Project - Build System"
	@echo "-------------------------------------------"
	@echo "make all       - Build both PDFs and HTML"
	@echo "make pdf       - Build PDFs only"
	@echo "make html      - Build HTML only (includes PDF linking)"
	@echo "make clean     - Remove LaTeX temp files"
	@echo "make distclean - Remove all generated output"
