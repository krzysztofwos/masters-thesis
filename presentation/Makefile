QUARTO_PYTHON := $(shell uv python find)
FSWATCH ?= $(shell command -v fswatch 2>/dev/null || command -v /opt/homebrew/bin/fswatch 2>/dev/null || command -v /usr/local/bin/fswatch 2>/dev/null)
INOTIFYWAIT ?= $(shell command -v inotifywait 2>/dev/null)
BIB ?= $(firstword $(wildcard ../references.bib ../thesis/references.bib))

.PHONY: all
all: html pdf

.PHONY: html
html: presentation.html

.PHONY: pdf
pdf: presentation.pdf

presentation.html: presentation.qmd $(wildcard slides/*.qmd) $(BIB) theme.scss
	@QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render presentation.qmd --to revealjs --output-dir .

presentation.pdf: presentation.html
	@npx decktape reveal $< $@ --chrome-arg=--allow-file-access-from-files --chrome-arg=--no-sandbox

.PHONY: clean
clean:
	@rm -rf presentation_files

.PHONY: dev
dev:
	@echo "Starting Quarto preview and file watcher..."
	@trap 'echo "Stopping services..."; kill 0' INT; \
	if [ -n "$(INOTIFYWAIT)" ]; then \
		(while inotifywait -e modify,create,delete -r slides/ --include='.*\\.qmd$$' 2>/dev/null; do \
			echo "Change detected in slides/"; \
			touch presentation.qmd; \
		done) & \
		(while inotifywait -e modify theme.scss 2>/dev/null; do \
			echo "Change detected in theme.scss, rebuilding..."; \
			QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render presentation.qmd --to revealjs --output-dir .; \
		done) & \
	elif [ -n "$(FSWATCH)" ]; then \
		echo "Using fswatch at $(FSWATCH)"; \
		($(FSWATCH) -o -r slides | \
		while IFS= read -r _; do \
			echo "$$(date +%H:%M:%S) Change detected in slides/"; \
			touch presentation.qmd; \
		done) & \
		($(FSWATCH) -o theme.scss | \
		while IFS= read -r _; do \
			echo "$$(date +%H:%M:%S) Change detected in theme.scss, rebuilding..."; \
			QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render presentation.qmd --to revealjs --output-dir .; \
		done) & \
	else \
		echo "Warning: no file watcher found. Install inotify-tools (Linux) or fswatch (macOS)."; \
	fi; \
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto preview presentation.qmd --to revealjs --host 0.0.0.0 --port 8001


.PHONY: distclean
distclean: clean
	@rm -f presentation.html presentation.pdf
