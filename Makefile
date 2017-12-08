# Requires apt packages pandoc and inkscape
report.pdf:
	make Advanced\ Lane\ Finding.pdf && cp Advanced\ Lane\ Finding.pdf report.pdf
Advanced\ Lane\ Finding.tex:
	jupyter nbconvert --to latex Advanced\ Lane\ Finding.ipynb 2>&1 | grep -v "Making directory Advanced Lane Finding_files"
Advanced\ Lane\ Finding.pdf:
	make Advanced\ Lane\ Finding.tex
	texfot --ignore '(Warning|Overfull|Underfull)' pdflatex  -interaction=nonstopmode -file-line-error Advanced\ Lane\ Finding.tex
	
view:
	okular report.pdf

.PHONY: Advanced\ Lane\ Finding.tex Advanced\ Lane\ Finding.pdf report.pdf
