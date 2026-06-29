# List of papers and their titles.
# The reason for dir:title split is the Latex ecosystem incapable to compose a set of papers
# from stand-alone papers. 
# This does not create root level PDF, only html.

PAPERS_LIST := \
	papers:Papers \
	book:Introduction_to_the_Theory \
#	ghghnd:Good_Heavens_Gods?_Hell_No,_Dogs!

NAME=abstract

# install directory
REMOTE_DIR := meskanen/$(NAME)

# Extract just the slugs/directories for target mapping
PAPER_DIRS := $(foreach p,$(PAPERS_LIST),$(word 1,$(subst :, ,$p)))

# Configurable Files
HEADER_TEX  = templates/header.tex
FOOTER_TEX  = templates/footer.tex
HEADER_HTML  = templates/header_html
FOOTER_HTML  = templates/footer_html
FINAL_TEX   = $(NAME).tex
FINAL_HTML  = index.html
DIST_DIR    = dist
TAR_FILE    = $(NAME).tar.gz

# packages to be installed (uploaded)
HTML_PACKAGES=$(TAR_FILE)
PDF_PACKAGES=

# 2. Define Phony Target Groups
# Creates lists like: pdf-introduction pdf-foo ...
PDF_TARGETS    := $(addprefix pdf-,$(PAPER_DIRS))
HTML_TARGETS   := $(addprefix html-,$(PAPER_DIRS))
UPLOAD_TARGETS := $(addprefix upload-,$(PAPER_DIRS))
CLEAN_TARGETS  := $(addprefix clean-,$(PAPER_DIRS))

.PHONY: all pdf html upload clean generate_tex generate_html package_html \
        $(PDF_TARGETS) $(HTML_TARGETS) $(UPLOAD_TARGETS) $(CLEAN_TARGETS)

# --- Global Umbrella Targets ---
all: pdf html
pdf: $(PDF_TARGETS)
html: $(HTML_TARGETS) $(FINAL_HTML) package_html
upload: $(UPLOAD_TARGETS)
clean: $(CLEAN_TARGETS) clean_root

include makes/upload.mak


# --- Pattern Rules for Sub-Paper Execution ---
# These dynamically match "pdf-slug" or "html-slug", switch into that dir, and run Make.
$(PDF_TARGETS): pdf-%:
	@echo "Building PDF for $*..."
	$(MAKE) -C $* pdf

$(HTML_TARGETS): html-%:
	@echo "Building HTML for $*..."
	$(MAKE) -C $* html

$(UPLOAD_TARGETS): upload-%:
	@echo "Uploading $*..."
	$(MAKE) -C $* upload

$(CLEAN_TARGETS): clean-%:
	@echo "Cleaning $*..."
	$(MAKE) -C $* clean


# --- Root Composition Targets ---

$(FINAL_TEX): $(HEADER_TEX) $(FOOTER_TEX)
	@echo "Generating intermediate dynamic LaTeX body..."
	@rm -f body.tex
	@for pair in $(PAPERS_LIST); do \
		slug=$$(echo "$$pair" | cut -d':' -f1); \
		title_raw=$$(echo "$$pair" | cut -d':' -f2); \
		title=$$(echo "$$title_raw" | tr '_' ' '); \
		label=$$(echo "$$slug" | tr -d '-'); \
		echo "% --- $$title ---" >> body.tex; \
		echo "\includepdf[pages=-, pagecommand={}, addtotoc={1, section, 1, {$$title}, $$label}]{$$slug/$$slug.pdf}" >> body.tex; \
	done
	@cat $(HEADER_TEX) body.tex $(FOOTER_TEX) > $(FINAL_TEX)
	@echo "Assembling master PDF..."


$(FINAL_HTML): $(HEADER_HTML) $(FOOTER_HTML)
	@echo "Generating HTML body..."
	@rm -f body.html
	@echo "<ul>" > body.html
	@for pair in $(PAPERS_LIST); do \
		slug=$$(echo "$$pair" | cut -d':' -f1); \
		title_raw=$$(echo "$$pair" | cut -d':' -f2); \
		title=$$(echo "$$title_raw" | tr '_' ' '); \
		echo "    <li><a href=\"$$slug/$$slug.html\">$$title</a></li>" >> body.html; \
	done
	@echo "</ul>" >> body.html
	@cat $(HEADER_HTML) body.html $(FOOTER_HTML) > $(FINAL_HTML)

package_html:
	@echo "Preparing distribution directory..."
	@rm -rf $(DIST_DIR) && mkdir -p $(DIST_DIR)
	@cp $(FINAL_HTML) $(DIST_DIR)/
	@for slug in $(PAPER_DIRS); do \
		if [ -d "$$slug/html-$$slug" ]; then \
			mkdir -p "$(DIST_DIR)/$$slug"; \
			cp -r "$$slug/html-$$slug/"* "$(DIST_DIR)/$$slug/"; \
		fi; \
	done
	@tar -czf $(TAR_FILE) -C $(DIST_DIR) .
	@echo "Successfully packed $(TAR_FILE)!"

clean_root:
	rm -f body.tex $(FINAL_TEX)  body.html $(FINAL_HTML) $(TAR_FILE)
	rm -rf $(DIST_DIR)
	rm $(NAME).*
	rm *.tar.gz */*.tar.gz

# Dependencies
