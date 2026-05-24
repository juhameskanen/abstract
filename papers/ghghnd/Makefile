FILENAME=ghghnd

# 1. Build the absolute Full Version (All chapters included)
pdf-full:
	latexmk -pdf -quiet -interaction=nonstopmode $(FILENAME).tex

# 2. Build the Physics-Only Version (Theology chapters completely stripped)
pdf-physics:
	latexmk -pdf -quiet -jobname=$(FILENAME)_physics -interaction=nonstopmode "\excludecomment{theology} \input{$(FILENAME).tex}"

# 3. Build the Theology-Only Version (Physics chapters completely stripped)
pdf-theology:
	latexmk -pdf -quiet -jobname=$(FILENAME)_theology -interaction=nonstopmode "\excludecomment{physics} \input{$(FILENAME).tex}"

# 4. Generate the Chunked HTML Website (Using the native compiler we discussed)
html-full:
	make4ht -c $(FILENAME).cfg -d html-output $(FILENAME).tex "xhtml,2,sections+,next"
	make4ht -m clean $(FILENAME).tex

clean:
	make4ht -m clean $(FILENAME).tex
