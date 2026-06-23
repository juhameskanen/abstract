
PAPERS_LIST := \
	introduction:Introduction \
	informational-derivation-of-reality:The_Axiomatic_Foundations_of_Reality \
	black-hole-singularities:Black_Hole_Singularities \
	emergence-of-spacetime:Emergent_Spacetime_Structures_from_Zero-Entropy_Initial_States \
	geometry-of-spacetime:The_Geometry_of_Spacetime_as_an_Information-Theoretic_Structure \
	emergence-of-gr:Space-time_Curvature_as_an_Information-Theoretic_Structure \
	wavefunction-as-compression:Wavefunction_as_Compression \
	informational-action:Classical_Geodesics_as_Minimal-Spectral-Complexity_Trajectories_in_Informational_Space \
	cosmological-model:Inflation,_Cosmological_and_Hubble_Constants \
	real-theory-of-everything:Real_Theory_of_Everything


# Extract just the slugs/directories for target mapping
PAPER_DIRS := $(foreach p,$(PAPERS_LIST),$(word 1,$(subst :, ,$p)))

# Configurable Files
HEADER_TEX  = header.tex
FOOTER_TEX  = footer.tex
FINAL_TEX   = aup.tex
FINAL_PDF   = aup.pdf
FINAL_HTML  = index.html
DIST_DIR    = dist
TAR_FILE    = aup.tar.gz

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
pdf: $(PDF_TARGETS) $(FINAL_PDF)
html: $(HTML_TARGETS) $(FINAL_HTML) package_html
upload: $(UPLOAD_TARGETS)
clean: $(CLEAN_TARGETS) clean_root

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

$(FINAL_PDF): $(FINAL_TEX)
	@latexmk -pdf -quiet -interaction=nonstopmode \
	    -pdflatex="pdflatex %O '$(PROF_MACRO)\input{%S}'" $(FINAL_TEX)




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


$(FINAL_HTML): header.html footer.html
	@echo "Generating HTML body..."
	@rm -f body.html
	@echo "<ul>" > body.html
	@for pair in $(PAPERS_LIST); do \
		slug=$$(echo "$$pair" | cut -d':' -f1); \
		title_raw=$$(echo "$$pair" | cut -d':' -f2); \
		title=$$(echo "$$title_raw" | tr '_' ' '); \
		echo "    <li><a href=\"$$slug/html-$$slug/$$slug.html\">$$title</a></li>" >> body.html; \
	done
	@echo "</ul>" >> body.html
	@cat header.html body.html footer.html > $(FINAL_HTML)

package_html:
	@echo "Preparing distribution directory..."
	@rm -rf $(DIST_DIR) && mkdir -p $(DIST_DIR)
	@cp $(FINAL_HTML) $(DIST_DIR)/
	@for slug in $(PAPER_DIRS); do \
		if [ -d "$$slug/html-$$slug" ]; then \
			mkdir -p "$(DIST_DIR)/$$slug"; \
			cp -r "$$slug/html-$$slug" "$(DIST_DIR)/$$slug/"; \
		fi; \
	done
	@tar -czf $(TAR_FILE) -C $(DIST_DIR) .
	@echo "Successfully packed $(TAR_FILE)!"

clean_root:
	rm -f body.tex $(FINAL_TEX) $(FINAL_PDF) body.html $(FINAL_HTML) $(TAR_FILE)
	rm -rf $(DIST_DIR)
