YEAR ?= 2026

.PHONY: season fetch site test clean

# end-to-end: fetch (cached), lineups, stints, RAPM, validation, site
season:
	python3 run.py --season $(YEAR)

fetch:
	Rscript R/fetch_data.R --season $(YEAR)

site:
	python3 -c "import sys; sys.path.insert(0, 'src'); from wbbrapm.site import build_site; build_site($(YEAR))"

test:
	python3 -m pytest tests -q

clean:
	rm -rf data/processed site
