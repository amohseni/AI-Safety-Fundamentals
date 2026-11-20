# Reader pilot: plan

Status: plan only (2026-11-19). Nothing here has been run. The owner decides who the readers are and when.

## Question

Does working through the core path teach a new graduate student enough to follow, and criticize, a current AI safety paper? The site's checks so far test the site (numbers, layout, behavior). They do not test whether anyone learns from it.

## Readers

Six to ten readers who match the intended reader: linear algebra, calculus, and probability, and no machine learning course. Six is enough to find the pages that fail most readers. It is not enough to estimate an average gain with any precision, and the report should say so.

## Procedure

1. **Before (20 minutes).** The pretest below, on paper or in a form, with no access to the site.
2. **Core path (about 5 hours, over one to two weeks).** Readers use the published site in their own browser. They are asked to keep a note of any page where they were stuck for more than five minutes.
3. **After (30 minutes), within two days of finishing.** The posttest, then the critique task.
4. **Two weeks later (15 minutes).** The delayed test, with no review beforehand.
5. **Interview (15 minutes).** Which page was least clear, which tool taught the most, what they would cut.

## Instruments

- **Pretest and posttest.** Two forms of twelve questions, one or two per section, each a new case in the style of the transfer questions in `docs/data/transfer.json` and not shown on the site. Half the readers get form A first and half form B, so a difference between the forms does not look like learning. These twenty-four questions have to be written and reviewed before the pilot; the eight transfer questions on the site cannot be reused, since readers will have seen them.
- **Critique task.** A two-page excerpt from a recent safety paper (candidate: the method and results of a linear-probe or activation-patching paper). The reader writes one paragraph on what the result shows and one on what it does not show. Scored by two people against a short rubric: names the method correctly; separates what was measured from what was concluded; names one alternative explanation.
- **Delayed test.** Six questions from the form the reader has not yet seen twice.
- **From the browser, with the reader's consent.** Readers export their saved progress (the `asf-v1` entry in the browser's storage): which checks were right at the first attempt, which were missed, and the self-explanations. The site sends nothing anywhere, so this needs an export button or a copy-and-paste step. Not built yet.

## What is reported

- Per reader: pretest, posttest, and delayed scores; time spent; critique rubric scores.
- Per page: share of readers whose first answer to each check was wrong; the pages readers reported being stuck on.
- The three pages with the worst first-attempt results, with the readers' own account of why.

## What would change the site

- A check that most readers miss on the first attempt, on a page they report as clear, means the check or the page teaches something other than what it tests. Rewrite one of them.
- A page most readers report being stuck on is rewritten or split.
- No gain on a section's posttest questions means the section's core pages do not carry its idea. That section is reviewed before any other change.

## Open decisions for the owner

1. Who the readers are (for example, incoming students in the department, or team members on current projects) and whether they are paid.
2. Whether this needs ethics review. If the results are only used to improve the site, it usually does not; if they are to be published, ask the review board first.
3. Whether to build the export button for saved progress.
