# Requires apt packages pandoc and inkscape
all:
	make Advanced\ Lane\ Finding.tex && \
	make Advanced\ Lane\ Finding.pdf
Advanced\ Lane\ Finding.tex:
	jupyter nbconvert --to latex Advanced\ Lane\ Finding.ipynb 2>&1 | grep -v "Making directory Advanced Lane Finding_files"
Advanced\ Lane\ Finding.pdf:
	texfot --ignore '(Warning|Overfull|Underfull)' pdflatex  -interaction=nonstopmode -file-line-error Advanced\ Lane\ Finding.tex
view:
	okular Advanced\ Lane\ Finding.pdf

.PHONY: Advanced\ Lane\ Finding.tex Advanced\ Lane\ Finding.pdf