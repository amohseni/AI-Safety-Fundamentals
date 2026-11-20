# A Primer on AI

An interactive introduction to how modern AI systems work, from a single neuron to the alignment problem, for a new graduate student teaching themselves. Each page answers one question with one interactive tool.

- The site lives in `docs/` and is published at https://aydinmohseni.com/teaching/primer-on-ai/ (copied there by the deploy workflow of `amohseni/Aydin-Mohseni-Website`). This repo's own GitHub Pages copy is a preview.
- To view it locally: `python3 -m http.server -d docs` and open http://localhost:8000.
- The page specifications are in `specs/`. Working rules for building pages are in `CLAUDE.md`.

## Licence

- Text and figures: CC BY 4.0 (`LICENSE-TEXT.md`).
- Code: MIT (`LICENSE`).
- Material from other sources keeps its own licence: the ET Book fonts and Tufte CSS (MIT); D3 (ISC), KaTeX (MIT), and js-tiktoken (MIT) in `docs/vendor/`; city facts from GeoNames (CC BY 4.0) in `docs/data/truth/`; and stored outputs and activations of the pretrained models named in `CLAUDE.md`, which are subject to those models' licences. Each `tools/<page-id>/README.md` names the model or data source its page uses.
