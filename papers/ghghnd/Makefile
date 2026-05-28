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
	rm $(FILENAME).glo $(FILENAME).ist $(FILENAME).bcf \
	$(FILENAME).blg $(FILENAME).aux $(FILENAME).bbl $(FILENAME).fls \
	$(FILENAME).toc $(FILENAME)_physics.* $(FILENAME).fdb_latexmk \
	$(FILENAME).run.xml

html-full:
	make4ht -c $(FILENAME).cfg -d html-output $(FILENAME).tex "xhtml,2,sections+,next"
	make4ht -m clean $(FILENAME).tex

