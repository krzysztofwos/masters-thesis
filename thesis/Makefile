# Makefile for UTokyo Creative Informatics Master's Thesis
# Uses platex (Japanese LaTeX) + dvipdfmx as required by cimt class

LATEX = platex
BIBTEX = pbibtex
DVIPDFMX = dvipdfmx
MAKEINDEX = makeindex

# Main document
TARGET = thesis
SOURCES = $(wildcard chapters/*.tex)
BIBFILE = references.bib

# Pattern for detecting label changes
LABPAT = "^LaTeX Warning: Label(s) may have changed\."

.PHONY: all
all: $(TARGET).pdf

.PHONY: help
help:
	@echo "Build targets:"
	@echo "  make all        # build thesis.pdf (default)"
	@echo "  make quick      # quick build without bibliography update"
	@echo ""
	@echo "Quality:"
	@echo "  make lint       # check for LaTeX issues (chktex)"
	@echo "  make fmt        # format .tex files in place (latexindent)"
	@echo "  make check      # lint + build"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      # remove intermediate files"
	@echo "  make distclean  # remove all generated files including PDF"
	@echo ""
	@echo "Utilities:"
	@echo "  make view       # open the PDF (macOS)"
	@echo ""
	@echo "Requirements: TeX Live with Japanese support (platex, pbibtex, dvipdfmx)"
	@echo "              chktex, latexindent (for lint/fmt)"

# Main compilation rule
$(TARGET).pdf: $(TARGET).dvi
	$(DVIPDFMX) $<

$(TARGET).dvi: $(TARGET).tex $(SOURCES) $(TARGET).bbl cimt.cls
	$(LATEX) $(TARGET)
	@while grep $(LABPAT) $(TARGET).log; do $(LATEX) $(TARGET); done

# Bibliography
$(TARGET).bbl: $(BIBFILE) $(TARGET).tex cimt.cls
	$(LATEX) $(TARGET)
	$(BIBTEX) $(TARGET)
	$(LATEX) $(TARGET)

.PHONY: quick
quick: $(TARGET).tex $(SOURCES) cimt.cls
	$(LATEX) $(TARGET)
	$(DVIPDFMX) $(TARGET)

.PHONY: clean
clean:
	$(RM) *.aux *.log *.dvi *.bbl *.blg *.toc *.lof *.lot *.out *.fls *.fdb_latexmk
	$(RM) chapters/*.aux

.PHONY: distclean
distclean: clean
	$(RM) $(TARGET).pdf

.PHONY: view
view: $(TARGET).pdf
	open $(TARGET).pdf

# Quality targets
.PHONY: lint
lint:
	@echo "Running chktex on thesis..."
	@chktex -q $(TARGET).tex 2>&1 || true
	@echo "Running chktex on chapters..."
	@for f in $(SOURCES); do chktex -q "$$f" 2>&1; done || true
	@echo "Lint complete."

.PHONY: fmt
fmt:
	@echo "Formatting thesis.tex..."
	@latexindent -w -s $(TARGET).tex
	@echo "Formatting chapters..."
	@for f in $(SOURCES); do latexindent -w -s "$$f"; done
	@$(RM) *.bak0 chapters/*.bak0 indent.log
	@echo "Format complete."

.PHONY: check
check: lint all
	@echo "Check complete."
