# A Primer on AI: working rules

Read this first in every session that works on the site.

## What this is

An interactive explainer site that teaches a new graduate student how modern AI systems work, from a single neuron to the alignment problem. It is used one-to-one for self-teaching: no grades, no accounts. Owner: Aydin Mohseni. Public repo, served by GitHub Pages from `docs/`.

## Where things are

| Path | What it holds |
| --- | --- |
| `specs/` | The page specifications, one file per page, and `specs/overview.md` (principles, how the site teaches, visual design, fixed terms, core path, build order). The specs are the source of truth for content. |
| `docs/` | The site itself. Static files, no build step. |
| `docs/pages/<id>.html` | One page, e.g. `docs/pages/2-5.html` for page 2.5. |
| `docs/js/page/<id>.js` | That page's tool code. |
| `docs/js/lib/` | Shared components: `mlp.js` (network, datasets), `plane.js` (plane view), `chart.js` (chart kit), `ui.js` (controls). |
| `docs/js/site.js` | Page shell: top line, previous/next, tasks, check questions, self-explanation boxes, progress. |
| `docs/js/pages.js` | Generated list of pages. Run `node tools/build-registry.mjs` after adding a page. |
| `docs/css/tufte.css`, `docs/css/site.css` | Tufte CSS (vendored, font paths edited) and the site layer with color tokens. |
| `tools/` | `check.mjs` (browser check and screenshots), `build-registry.mjs`, `test-*.mjs` (numerical tests behind claims in the text). |

## How a page is built

1. Read the page's spec in `specs/` and the overview sections it depends on.
2. Copy the structure of the reference pages, `docs/pages/2-5.html` and `docs/pages/2-4.html`: title, text with sidenotes for references, the static figure, the tool, the task list, check questions, the self-explanation box, the takeaway.
3. Every task asks for a prediction before the reader can reveal what to notice (`.task` markup is documented in `docs/js/site.js`).
4. Any claim a task or notice makes about what the tool will show must be tested numerically first (see `tools/test-mlp.mjs`). If a claim holds only in most runs, say so and give the numbers.
5. Run `node tools/build-registry.mjs`, then `node tools/check.mjs <id>`; look at the desktop and phone screenshots.
6. Have an agent that did not build the page review it against the checklist below before showing it to Aydin.
7. Aydin reviews by section, using the Review column of the planning doc.

## Review checklist

- Accuracy: every claim, number, and formula verified; references real and correctly cited.
- Pedagogy: one idea per page; concrete case before the general statement; prediction before every reveal; the check question tests the page's aim.
- Writing: plain words; no em dashes; no clever hooks or rhetorical contrasts; clear referents; the terms in the Fixed terms table of `specs/overview.md`, used the same way everywhere.
- Design: no chart frames, grids, legends, shadows, or gradients; labels on the lines; one accent color for what the reader manipulates, one second color only when two quantities are compared; works at 375 px; keyboard operable; respects reduced motion; no horizontal overflow.
- Code: no console errors; the tool survives switching any control at any time.

## Conventions

- Three type sizes only (decided 2026-10-15): the page title (h1), section headings (h2), and 16pt (`--text`) for everything else, including chart labels, controls, sidenotes, and formulas. Gray and italics carry other distinctions. Charts are drawn at their true pixel width so their labels stay at 16pt.
- Colors only through the CSS tokens in `site.css` (`--accent`, `--second`, `--ink`, `--quiet`, `--rule`), so light and dark themes both work.
- Libraries are vendored in `docs/vendor/` at fixed versions (D3 7.9.0, KaTeX 0.16.11). No CDNs.
- Reader state lives in `localStorage` under one key through `store` in `site.js`; every access tolerates failure.
- Page files are named by page number with a dash: page 3.4 is `3-4`.

## Status

Built and independently reviewed (2026-10-17): 38 pages: 0.1, 1.1 to 1.5, 2.1 to 2.8, 3.1, 3.4, 3.8, 4.2, 4.3, 5.1 to 5.6, 6.1, 6.4, 6.5, 6.7, 7.1, 7.3, 8.1 to 8.6, 9.1. Each spec ends with the builder's changes and the reviewer's changes. `node tools/build-registry.mjs` lists what is built.

Waiting on pretrained model data (huggingface.co is unreachable from the build workspace): 1.6, 3.2, 3.3, 3.5 to 3.7, 4.1, 4.4, 6.2, 6.3, 6.6, 7.2. Page 6.7 runs live activation patching on the modular-addition model instead of GPT-2.

Modular addition runs use p = 59, not Nanda et al.'s 113, so training fits on a CPU (`tools/modadd/README.md`).

Sidenote numbers are set by `site.js` (`data-n`), because the Tufte CSS counter showed every note as 1.
