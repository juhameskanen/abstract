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

# -- Flattened paper list (full slugs) --

PAPERS := $(shell jq -r '[.sections[].items[] as $$item | $$item.slug as $$slug | (if ($$item.children? | length > 0) then $$item.children[] | "\($$slug)/\(.slug)" else $$slug end)] | .[]' papers.json)

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
	@SLUG_DIR=$(PAPERS_DIR)/$(PAPER); \
	if [ ! -d "$$SLUG_DIR" ]; then echo "Folder $$SLUG_DIR does not exist"; exit 1; fi; \
	find "$$SLUG_DIR" -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r tex; do \
		echo "Compiling $$tex"; \
		dir=$$(dirname "$$tex"); \
		base=$$(basename "$$tex" .tex); \
		cd "$$dir"; \
		$(LATEXMK) "$$base.tex"; \
		# Flatten PDF filename: replace / with - in slug \
		safe_name=$$(echo "$(PAPER)" | sed 's|/|-|g'); \
		cp "$$base.pdf" "$(ROOT_DIR)/$(PDF_OUT)/$$safe_name-$$base.pdf"; \
		cd - > /dev/null; \
	done

# -- HTML BUILD --

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
	@echo '</body></html>' >> $(HTML_OUT)/index.html

html-index:
	@PROJECT_TITLE=$$(jq -r '.title' papers.json); \
	echo "<html><head><title>$$PROJECT_TITLE</title>" \
	     "<meta name='color-scheme' content='light dark'>" \
	     "<link rel='stylesheet' href='style.css'></head>" \
	     "<body class='bodytext'><h1>$$PROJECT_TITLE</h1>" > $(HTML_OUT)/index.html; \
	# Loop over sections \
	jq -c '.sections[]' papers.json | while read -r section; do \
		section_title=$$(echo $$section | jq -r '.title'); \
		echo "<h2>$$section_title</h2>" >> $(HTML_OUT)/index.html; \
		items=$$(echo $$section | jq -c '.items'); \
		$(MAKE) html-list ITEMS="$$items" PARENT=""; \
	done

# Recursive function to handle hierarchy
html-list:
	@echo "$$ITEMS" | jq -c '.[]' | while read -r item; do \
		title=$$(echo $$item | jq -r '.title'); \
		slug=$$(echo $$item | jq -r '.slug'); \
		full_path="$$( [ -z "$(PARENT)" ] && echo "$$slug" || echo "$(PARENT)/$$slug" )"; \
		html_dir="$(HTML_OUT)/$$full_path"; \
		mkdir -p "$$html_dir"; \
		# Copy PDF if exists \
		PDF_FILE=$$(find $(PDF_OUT) -type f -name "*$$(echo "$$full_path" | sed 's|/|-|g')*.pdf" | head -n1); \
		PDF_LINK=""; \
		[ -n "$$PDF_FILE" ] && PDF_LINK=" | <a href='./$$(basename "$$PDF_FILE")'>[PDF]</a>" && cp "$$PDF_FILE" "$$html_dir/"; \
		# Convert .tex files \
		TEX_DIR="$(PAPERS_DIR)/$$full_path"; \
		if [ -d "$$TEX_DIR" ]; then \
			find "$$TEX_DIR" -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r tex; do \
				base=$$(basename "$$tex" .tex); \
				out="$$html_dir/index.html"; \
				css_rel=$$(realpath --relative-to="$$html_dir" "$(STYLE_CSS)"); \
				[ -d "$$TEX_DIR/figures" ] && cp -r "$$TEX_DIR/figures" "$$html_dir/"; \
				(cd "$$(dirname "$$tex")" && $(PANDOC) "$$base.tex" --bibliography="$(BIB_FILE)" -c "$$css_rel" -o "$$out"); \
			done; \
		fi; \
		# Write link to index \
		echo "<li><a href='./$$full_path/'>$$title</a>$$PDF_LINK" >> $(HTML_OUT)/index.html; \
		# Process children recursively \
		children=$$(echo $$item | jq -c '.children'); \
		if [ "$$(echo "$$children" | jq 'length')" -gt 0 ]; then \
			$(MAKE) html-list ITEMS="$$children" PARENT="$$full_path"; \
		fi; \
		echo "</li>" >> $(HTML_OUT)/index.html; \
	done

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
