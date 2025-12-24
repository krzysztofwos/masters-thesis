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

THISDIR = sampleE
BASE1=sampleE
BASE2=sampleEP
CLS=cimt.cls
CLSORIG=../class/$(CLS)
SRCS=eabst.tex jabst.tex intro.tex introP.tex body.tex concl.tex publications.tex ack.tex appendix.tex

bases = $(BASE1) $(BASE2)
targets = $(bases:%=%.pdf)
bbls = $(bases:%=%.bbl)

DISTDIR = ../dist
DISTPDFS = $(targets)
DISTTXTS = $(SRCS) $(bases:%=%.tex) $(bases:%=%.bib) Makefile
DISTBINS = fig.eps

UTILDIR = ../../util
CPWEB = $(UTILDIR)/todist -t web -d $(DISTDIR) -s doc
CPTXT = $(UTILDIR)/todist -t txt -d $(DISTDIR) -s $(THISDIR)
CPBIN = $(UTILDIR)/todist -t bin -d $(DISTDIR) -s $(THISDIR)

.PHONY: all clean dist-pdf dist-txt dist-bin

all: $(targets)

$(CLS): $(CLSORIG)
	cp $(CLSORIG) $(CLS)

$(BASE1).dvi: $(BASE1).bbl $(SRCS) $(CLS)

$(BASE2).dvi: $(BASE2).bbl $(SRCS) $(CLS)

$(bbls): $(CLS)

clean:
	$(RM) *~ *.log *.dvi *.pdf *.aux *.bbl *.blg *.toc *.flc *.bak *.bmc $(CLS) fig.pbm

dist: dist-pdf dist-txt dist-bin

dist-pdf: $(DISTPDFS)
	$(CPWEB) $(DISTPDFS)

dist-txt: $(DISTTXTS)
	$(CPTXT) $(DISTTXTS)

dist-bin: $(DISTBINS)
	$(CPBIN) $(DISTBINS)
