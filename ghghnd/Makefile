DOCS       := ghghnd
REMOTE_DIR := meskanen/abstract/ghghnd
TAR_FILE    = $(DOCS).tar.gz

# html package, for uploading
HTML_PACKAGES = $(TAR_FILE)
PDF_PACKAGES=$(DOCS).pdf


include ../makes/abstract.mak
include ../makes/upload.mak

all: package_html

package_html:
	@cp html-$(DOCS)/$(DOCS).html html-$(DOCS)/index.html
	@tar -czf $(TAR_FILE) -C html-$(DOCS) .
	@echo "Successfully packed $(TAR_FILE)!"
