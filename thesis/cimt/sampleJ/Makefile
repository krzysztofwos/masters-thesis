LATEX=platex
MAKEINDEX=makeindex
BIBTEX=pbibtex
LABPAT="^LaTeX Warning: Label(s) may have changed\."

%.pdf: %.dvi
	dvipdfmx $<

%.dvi: %.tex
	$(LATEX) $*
	while grep $(LABPAT) $*.log; do $(LATEX) $<; done

%.bbl: %.bib
	$(LATEX) $*
	$(BIBTEX) $*
	$(LATEX) $*

THISDIR = sampleJ
BASE=sampleJ
CLS=cimt.cls
CLSORIG=../class/$(CLS)
SRCS = jabst.tex eabst.tex intro.tex body.tex concl.tex publications.tex ack.tex appendix.tex
DISTDIR = ../dist
DISTPDFS = $(BASE).pdf
DISTTXTS = $(SRCS) $(BASE).tex $(BASE).bib Makefile 
DISTBINS = fig.eps 
UTILDIR = ../../util
CPWEB = $(UTILDIR)/todist -t web -d $(DISTDIR) -s doc
CPTXT = $(UTILDIR)/todist -t txt -d $(DISTDIR) -s $(THISDIR)
CPBIN = $(UTILDIR)/todist -t bin -d $(DISTDIR) -s $(THISDIR)

.PHONY: all clean dist dist-pdf dist-txt dist-bin

all: $(BASE).pdf

$(CLS): $(CLSORIG)
	cp $(CLSORIG) $(CLS)

$(BASE).dvi: $(BASE).bbl $(SRCS) $(CLS)

$(BASE).bbl: $(CLS)

clean:
	$(RM) *~ *.log *.dvi *.pdf *.aux *.bbl *.blg *.toc *.flc *.bak *.bmc $(CLS) fig.pbm

dist: dist-pdf dist-txt dist-bin

dist-pdf: $(DISTPDFS)
	$(CPWEB) $(DISTPDFS)

dist-txt: $(DISTTXTS)
	$(CPTXT) $(DISTTXTS)

dist-bin: $(DISTBINS)
	$(CPBIN) $(DISTBINS)
