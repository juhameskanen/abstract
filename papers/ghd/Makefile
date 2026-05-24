
# 1. Build the absolute Full Version (All chapters included)
pdf-full:
	latexmk -pdf -quiet -interaction=nonstopmode ghd.tex

# 2. Build the Physics-Only Version (Theology chapters completely stripped)
pdf-physics:
	latexmk -pdf -quiet -jobname=ghd_physics -interaction=nonstopmode "\excludecomment{theology} \input{ghd.tex}"

# 3. Build the Theology-Only Version (Physics chapters completely stripped)
pdf-theology:
	latexmk -pdf -quiet -jobname=ghd_theology -interaction=nonstopmode "\excludecomment{physics} \input{ghd.tex}"

# 4. Generate the Chunked HTML Website (Using the native compiler we discussed)
html-full:
	make4ht -c ghd.cfg -d html-output $(FLAGS) ghd.tex "xhtml,1,sections+"
	make4ht -m clean ghd.tex
