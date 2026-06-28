REMOTE_DIR := meskanen/book
DOCS       := book
TAR_FILE    = book.tar.gz

# html package, for uploading
HTML_PACKAGES = $(TAR_FILE)
PDF_PACKAGES=book.pdf


include ../makes/abstract.mak
include ../makes/upload.mak

all: package_html

package_html:
	@cp html-book/book.html html-book/index.html
	@tar -czf $(TAR_FILE) -C html-$(DOCS) .
	@echo "Successfully packed $(TAR_FILE)!"
