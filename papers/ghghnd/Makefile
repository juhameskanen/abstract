FILENAME=ghghnd

pdf-full:
	latexmk -pdf -quiet \
	-interaction=nonstopmode \
	$(FILENAME).tex

pdf-physics:
	latexmk -pdf -quiet \
	-jobname=$(FILENAME)_physics \
	-pdflatex='pdflatex %O "\def\buildtheology{false}\input{%S}"' \
	$(FILENAME).tex

pdf-theology:
	latexmk -pdf -quiet \
	-jobname=$(FILENAME)_theology \
	-pdflatex='pdflatex %O "\def\buildphysics{false}\input{%S}"' \
	$(FILENAME).tex

clean:
	latexmk -C

html-full:
	make4ht -c $(FILENAME).cfg -d html-output $(FILENAME).tex "xhtml,2,sections+,next"
	make4ht -m clean $(FILENAME).tex

clean:
	make4ht -m clean $(FILENAME).tex
