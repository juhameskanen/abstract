SHELL := /bin/bash

# -- Configuration --

ROOT_DIR := $(shell pwd)
PAPERS_DIR := papers
PDF_OUT := pdf-output
HTML_OUT := html
BIB_FILE := $(ROOT_DIR)/papers/references.bib
STYLE_CSS := style.css

PAPERS := $(shell jq -r '.[]' papers.json)

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error
PANDOC := pandoc -s -M ishtml=true --from=latex --to=html5 --citeproc \
          --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js

# -- Phony Targets -- 

.PHONY: all pdf html clean distclean list-papers

all: pdf html

list-papers:
	@echo "$(PAPERS)"


# -- PDF BUILD (equivalent to build-pdf.yml) --

pdf: $(PDF_OUT)
	@for paper in $(PAPERS); do \
		echo "==> Building PDFs for $$paper"; \
		$(MAKE) pdf-paper PAPER=$$paper; \
	done

$(PDF_OUT):
	mkdir -p $(PDF_OUT)

pdf-paper:
	@cd $(PAPERS_DIR)/$(PAPER) && \
	find . -name "*.tex" ! -name "_*" | while read tex; do \
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


# -- HTML BUILD (equivalent to build-pages.yml) --


html: pdf
	rm -rf $(HTML_OUT)
	mkdir -p $(HTML_OUT)
	touch $(HTML_OUT)/.nojekyll
	cp $(STYLE_CSS) $(HTML_OUT)/style.css
	$(MAKE) html-index
	@for paper in $(PAPERS); do \
		echo "==> Building HTML for $$paper"; \
		$(MAKE) html-paper PAPER=$$paper; \
	done
	@echo '</ul></body></html>' >> $(HTML_OUT)/index.html

html-index:
	@echo '<html><head><title>The Abstract Universe Theory (AUT)</title>' \
	      '<meta name="color-scheme" content="light dark">' \
	      '<link rel="stylesheet" href="style.css"></head>' \
	      '<body class="bodytext"><h1>The Abstract Universe Theory (AUT)</h1>' \
	      '<h2>Research Papers</h2><ul>' \
	      > $(HTML_OUT)/index.html

html-paper:
	@paper=$(PAPER); \
	html_paper_dir=$(HTML_OUT)/$$paper; \
	mkdir -p $$html_paper_dir; \
	cp $(PDF_OUT)/*$$paper*.pdf $$html_paper_dir/ 2>/dev/null || true; \
	echo "" > supp_list.tmp; \
	echo "MAIN_PDF=''" > main_pdf.tmp; \
	find $(PAPERS_DIR)/$$paper -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r tex; do \
		base=$$(basename "$$tex" .tex); \
		rel=$${tex#$(PAPERS_DIR)/$$paper/}; \
		dir_abs=$$(cd "$$(dirname "$$tex")" && pwd); \
		if [ "$$rel" = "$$base.tex" ]; then \
			out="$(ROOT_DIR)/$$html_paper_dir/index.html"; \
			css="../style.css"; \
			[ -d "$(PAPERS_DIR)/$$paper/figures" ] && cp -r "$(PAPERS_DIR)/$$paper/figures" "$$html_paper_dir/"; \
			echo "MAIN_PDF=' | <a href=\"./$$paper/$$base.pdf\">[PDF]</a>'" > main_pdf.tmp; \
		else \
			outdir="$(ROOT_DIR)/$$html_paper_dir/$$base"; \
			out="$$outdir/index.html"; \
			mkdir -p "$$outdir"; \
			css="../../style.css"; \
			[ -d "$$dir_abs/figures" ] && cp -r "$$dir_abs/figures" "$$outdir/"; \
			echo "<li><a href='./$$paper/$$base/'>$$base</a> | <a href='./$$paper/supplementary-$$base.pdf'>[PDF]</a></li>" >> supp_list.tmp; \
		fi; \
		echo "Converting $$tex..."; \
		(cd "$$dir_abs" && $(PANDOC) "$$(basename "$$tex")" \
			--bibliography="$(BIB_FILE)" \
			-c "$$css" \
			-o "$$out") || exit 1; \
	done || exit 1; \
	SUPP_LIST=$$(cat supp_list.tmp); \
	source main_pdf.tmp; \
	if [ -n "$$SUPP_LIST" ]; then \
		echo "<li><a href='./$$paper/'>$$paper</a>$$MAIN_PDF<ul>$$SUPP_LIST</ul></li>" >> $(HTML_OUT)/index.html; \
	else \
		echo "<li><a href='./$$paper/'>$$paper</a>$$MAIN_PDF</li>" >> $(HTML_OUT)/index.html; \
	fi; \
	rm -f supp_list.tmp main_pdf.tmp


# -- Cleanup --

clean:
	find $(PAPERS_DIR) -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.bcf" \
		-o -name "*.blg" -o -name "*.log" -o -name "*.out" \
		-o -name "*.run.xml" -o -name "*.toc" \) -delete

distclean: clean
	rm -rf $(PDF_OUT) $(HTML_OUT)

# -- Install build tools -- 

prequisities:
	sudo apt-get install jq pandoc latexmk biber
