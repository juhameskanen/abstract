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

# Flatten all slugs from papers.json (including children)
PAPERS := $(shell jq -r '[.sections[].items[], .sections[].items[]?.children[]?] | .[].slug' papers.json)

# -- Phony Targets -- 

.PHONY: all pdf html clean distclean list-papers

all: pdf html

list-papers:
	@echo "$(PAPERS)"

# -- PDF BUILD --

pdf: $(PDF_OUT)
	@for paper in $(PAPERS); do \
		echo "==> Building PDFs for $$paper"; \
		$(MAKE) pdf-paper PAPER=$$paper; \
	done

$(PDF_OUT):
	mkdir -p $(PDF_OUT)

pdf-paper:
	@cd $(PAPERS_DIR)/$(PAPER) && \
	find . -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read tex; do \
		echo "Compiling $$tex"; \
		dir=$$(dirname "$$tex"); \
		base=$$(basename "$$tex" .tex); \
		cd "$$dir"; \
		$(LATEXMK) "$$(basename "$$tex")"; \
		if [ "$$dir" = "." ]; then \
			dest="$$base.pdf"; \
		else \
			clean=$$(echo "$$dir" | sed 's|./||; s|/|-|g'); \
			dest="$$clean-$$base.pdf"; \
		fi; \
		cp "$$base.pdf" "$(ROOT_DIR)/$(PDF_OUT)/$$dest"; \
		cd - > /dev/null; \
	done

# -- HTML BUILD --

html: pdf
	rm -rf $(HTML_OUT)
	mkdir -p $(HTML_OUT)
	touch $(HTML_OUT)/.nojekyll
	cp $(STYLE_CSS) $(HTML_OUT)/style.css
	$(MAKE) html-index
	@# Build HTML for each slug
	@for paper in $(PAPERS); do \
		echo "==> Building HTML for $$paper"; \
		$(MAKE) html-paper PAPER=$$paper; \
	done
	@echo '</body></html>' >> $(HTML_OUT)/index.html

html-index:
	@PROJECT_TITLE="$$(jq -r '.title' papers.json)"; \
	echo "<html><head><title>$$PROJECT_TITLE</title>" \
	     "<meta name='color-scheme' content='light dark'>" \
	     "<link rel='stylesheet' href='style.css'></head>" \
	     "<body class='bodytext'><h1>$$PROJECT_TITLE</h1>" \
	     > $(HTML_OUT)/index.html; \
	echo "" >> $(HTML_OUT)/index.html

html-paper:
	@paper=$(PAPER); \
	html_dir=$(HTML_OUT)/$$paper; \
	mkdir -p $$html_dir; \
	cp $(PDF_OUT)/*$$paper*.pdf $$html_dir/ 2>/dev/null || true; \
	TEX_DIR=$(PAPERS_DIR)/$$paper; \
	if [ -d "$$TEX_DIR" ]; then \
		find "$$TEX_DIR" -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r tex; do \
			base=$$(basename "$$tex" .tex); \
			out="$${html_dir}/index.html"; \
			css="../style.css"; \
			[ -d "$$TEX_DIR/figures" ] && cp -r "$$TEX_DIR/figures" "$$html_dir/"; \
			echo "Converting $$tex..."; \
			(cd "$$(dirname "$$tex")" && $(PANDOC) "$$(basename "$$tex")" \
				--bibliography="$(BIB_FILE)" \
				-c "$$css" \
				-o "$$out") || exit 1; \
		done; \
	fi; \
	TITLE=$$(jq -r --arg slug "$$paper" '[.sections[].items[], .sections[].items[]?.children[]?] | .[] | select(.slug == $$slug) | .title' papers.json); \
	PDF_FILE=$$(ls $$html_dir/*.pdf 2>/dev/null | head -n1); \
	PDF_LINK=""; \
	[ -n "$$PDF_FILE" ] && PDF_LINK=" | <a href='./$$(basename "$$PDF_FILE")'>[PDF]</a>"; \
	echo "<li><a href='./$$paper/'>$$TITLE</a>$$PDF_LINK</li>" >> $(HTML_OUT)/index.html

# -- Cleanup --

clean:
	find $(PAPERS_DIR) -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.bcf" \
		-o -name "*.blg" -o -name "*.log" -o -name "*.out" \
		-o -name "*.run.xml" -o -name "*.toc" \) -delete

distclean: clean
	rm -rf $(PDF_OUT) $(HTML_OUT)

# -- Install build tools -- 

prerequisites:
	sudo apt-get install jq pandoc latexmk biber
