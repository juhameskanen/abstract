SHELL := /bin/bash

# -- Configuration --

ROOT_DIR := $(shell pwd)
PAPERS_DIR := papers
PDF_OUT := pdf-output
HTML_OUT := html
BIB_FILE := $(ROOT_DIR)/papers/references.bib
STYLE_CSS := style.css

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error
PANDOC := pandoc -s -M ishtml=true --from=latex --to=html5 --citeproc \
          --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js

# -- Phony Targets --

.PHONY: all pdf html clean distclean list-papers

all: pdf html

# ------------------------------------------------------------
# PDF BUILD
# ------------------------------------------------------------

pdf: $(PDF_OUT)
	@echo "Building PDFs..."
	@jq -r '[.sections[].items[] as $$item | $$item.slug as $$slug | (if ($$item.children? | length > 0) then $$item.children[] | "\($$slug)/\(.slug)" else $$slug end)] | .[]' papers.json | while read -r SLUG; do \
		echo "==> Building PDFs for $$SLUG"; \
		$(MAKE) pdf-paper PAPER="$$SLUG"; \
	done

$(PDF_OUT):
	mkdir -p $(PDF_OUT)

pdf-paper:
	@SLUG=$(PAPER); \
	FULL_DIR=$(PAPERS_DIR)/$$SLUG; \
	if [ ! -d "$$FULL_DIR" ]; then echo "ERROR: Folder $$FULL_DIR not found"; exit 1; fi; \
	find "$$FULL_DIR" -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r tex; do \
		echo "Compiling $$tex"; \
		DIR_PATH=$$(dirname "$$tex"); \
		BASE_NAME=$$(basename "$$tex" .tex); \
		cd "$$DIR_PATH"; \
		$(LATEXMK) "$$BASE_NAME.tex"; \
		SAFE_SLUG=$$(echo "$$SLUG" | sed 's|/|-|g'); \
		DEST_NAME="$$SAFE_SLUG-$$BASE_NAME.pdf"; \
		cp "$$BASE_NAME.pdf" "$(ROOT_DIR)/$(PDF_OUT)/$$DEST_NAME"; \
		cd - > /dev/null; \
	done

# ------------------------------------------------------------
# HTML BUILD
# ------------------------------------------------------------

html: pdf
	rm -rf $(HTML_OUT)
	mkdir -p $(HTML_OUT)
	touch $(HTML_OUT)/.nojekyll
	cp $(STYLE_CSS) $(HTML_OUT)/

	@# Start index.html
	TITLE=$$(jq -r '.title' papers.json); \
	echo "<html><head><title>$$TITLE</title><meta name='color-scheme' content='light dark'><link rel='stylesheet' href='style.css'></head><body class='bodytext'><h1>$$TITLE</h1>" > $(HTML_OUT)/index.html

	@# Recursive function
	process_items() { \
		ITEMS_JSON="$1"; \
		PARENT_SLUG="$2"; \
		echo "<ul>" >> $(HTML_OUT)/index.html; \
		echo "$$ITEMS_JSON" | jq -c '.[]' | while read -r item; do \
			TITLE=$$(echo "$$item" | jq -r '.title'); \
			SLUG=$$(echo "$$item" | jq -r '.slug'); \
			if [ -z "$$PARENT_SLUG" ]; then \
				FULL_SLUG="$$SLUG"; \
			else \
				FULL_SLUG="$$PARENT_SLUG/$$SLUG"; \
			fi; \
			HTML_DIR="$(HTML_OUT)/$$FULL_SLUG"; \
			mkdir -p "$$HTML_DIR"; \
			TEX_DIR="$(PAPERS_DIR)/$$FULL_SLUG"; \
			SAFE_SLUG=$$(echo "$$FULL_SLUG" | sed 's|/|-|g'); \
			PDF_FILE=$$(find $(PDF_OUT) -type f -name "*$$SAFE_SLUG*.pdf" | head -n1); \
			PDF_LINK=""; \
			if [ -n "$$PDF_FILE" ]; then \
				cp "$$PDF_FILE" "$$HTML_DIR/"; \
				PDF_BASENAME=$$(basename "$$PDF_FILE"); \
				PDF_LINK=" | <a href='./$$PDF_BASENAME'>[PDF]</a>"; \
			fi; \
			# Convert all .tex files if folder exists \
			if [ -d "$$TEX_DIR" ]; then \
				find "$$TEX_DIR" -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r tex; do \
					BASE=$$(basename "$$tex" .tex); \
					[ -d "$$TEX_DIR/figures" ] && cp -r "$$TEX_DIR/figures" "$$HTML_DIR/"; \
					CSS_REL=$$(realpath --relative-to="$$HTML_DIR" "$(ROOT_DIR)/style.css"); \
					mkdir -p "$$HTML_DIR"; \
					$(PANDOC) "$$tex" -o "$$HTML_DIR/index.html" --bibliography="$(BIB_FILE)" -c "$$CSS_REL"; \
				done; \
			fi; \
			echo "<li><a href='./$$FULL_SLUG/'>$$TITLE</a>$$PDF_LINK" >> $(HTML_OUT)/index.html; \
			CHILDREN=$$(echo "$$item" | jq -c '.children // []'); \
			if [ "$$(echo "$$CHILDREN" | jq 'length')" -gt 0 ]; then \
				process_items "$$CHILDREN" "$$FULL_SLUG"; \
			fi; \
			echo "</li>" >> $(HTML_OUT)/index.html; \
		done; \
		echo "</ul>" >> $(HTML_OUT)/index.html; \
	}; \
	# Run for each top-level section \
	jq -c '.sections[]' papers.json | while read -r section; do \
		SECTION_TITLE=$$(echo "$$section" | jq -r '.title'); \
		echo "<h2>$$SECTION_TITLE</h2>" >> $(HTML_OUT)/index.html; \
		ITEMS=$$(echo "$$section" | jq -c '.items'); \
		process_items "$$ITEMS" ""; \
	done

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

clean:
	find $(PAPERS_DIR) -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.bcf" \
		-o -name "*.blg" -o -name "*.log" -o -name "*.out" \
		-o -name "*.run.xml" -o -name "*.toc" \) -delete

distclean: clean
	rm -rf $(PDF_OUT) $(HTML_OUT)

# ------------------------------------------------------------
# Install build tools
# ------------------------------------------------------------

prerequisites:
	sudo apt-get install jq pandoc latexmk biber
