# Link audit findings (branch audit-links, 2026-11-19)

Every reference in the sidenotes of `docs/pages/*.html` was checked against a primary record before it was linked: the arXiv API for arXiv ids, the Crossref API for DOIs, the NeurIPS, PMLR, JMLR, ACL Anthology, and OpenReview listings for proceedings papers, and the page itself for blog posts and reports. For each one the title and first author were compared with the citation.

No citation text was changed. The audit found no wrong year, venue, author, volume, or page range. The items below are the places where the record and the citation differ in some smaller way, or where the link does not point at exactly the edition cited.

## The cited source has moved or changed

1. **Page 1.2, note 6, NSF, "The Mansfield Amendment" (NSF history, nsf.gov).** The web page this most likely refers to, `www.nsf.gov/nsb/documents/2000/nsb00215/nsb50/1970/mansfield.html`, now returns 404. The same text is a section headed "The Mansfield Amendment" in the National Science Board report *The National Science Board: A History in Highlights 1950-2000* (NSB 00-215), whose PDF is live at `nsf-gov-resources.nsf.gov/nsb/documents/2000/nsb00215/nsb00215.pdf` (checked: the section is in the PDF text). The link points to that PDF. The Internet Archive reports a snapshot of the old page (2025-07-02), which could not be opened from this environment. Suggested change, not made: name the report in the citation.

2. **Page 7.3, note 2, Goldstein and Lederman (2025), "What does ChatGPT want? An interpretationist guide," PhilArchive.** Crossref now lists the paper as published: *Inquiry*, online 2026, pp. 1-31, DOI 10.1080/0020174X.2026.2727584, same title and authors. The link points to that DOI. The PhilArchive record could not be opened (the site blocks scripts), so the 2025 PhilArchive citation itself was not checked. Suggested change, not made: cite the *Inquiry* version, after checking that the passages page 7.3 relies on are unchanged.

3. **Page 1.5, note 7, Epoch AI, "Notable AI Models" (epoch.ai/data/notable-ai-models).** The address still works and now redirects to `epoch.ai/data/ai-models`, a page titled "Data on AI models" that contains the Notable AI models dataset. The link uses the address as cited.

4. **Page 6.6, note 3, Anthropic (2024), "Golden Gate Claude."** `anthropic.com/news/golden-gate-claude` redirects to `anthropic.com/research/golden-gate-claude`. The link uses the `/news/` address. The date May 23, 2024 is on the page.

## Small differences between the record and the citation

5. **Page 1.2, note 3, Block (1962).** The article's full title is "The Perceptron: A Model for Brain Functioning. I" (Crossref, DOI 10.1103/RevModPhys.34.123). The citation leaves out the "I".

6. **Page 6.1, note 1, Bereska and Gavves (2024).** OpenReview gives the title as "Mechanistic Interpretability for AI Safety - A Review", with a dash where the citation has a colon.

7. **Page 6.5, note 1, Cunningham et al. (2023), ICLR 2024.** The author order differs by version: arXiv 2309.08600 lists Cunningham, Ewart, Riggs, Huben, Sharkey; the ICLR 2024 record on OpenReview lists Huben first. "Cunningham et al. (2023)" matches the arXiv version, and the link goes to the ICLR record the citation names.

8. **Page 3.4, note 1, 3Blue1Brown (2024), "Attention in transformers, visually explained."** That is the title of the video. The lesson page on 3blue1brown.com, which the link points to, is titled "Attention in transformers, step-by-step" and is marked Chapter 6, as cited.

9. **Page 8.1, note 2, Strathern (1997).** Crossref's primary DOI for this article is an old Wiley one, 10.1002/(SICI)1234-981X(199707)5:3<305::AID-EURO184>3.0.CO;2-4. The link uses the Cambridge DOI 10.1017/S1062798700002660, which resolves to the article on Cambridge Core (title, author, volume 5, pp. 305-321 confirmed). The page reference "p. 308" was not checked.

## Papers cited as preprints that also have a published version

The pages cite these by arXiv id, so the links go to arXiv. Listed in case the citations should name the published version.

10. **Pages 1.1 and 1.5, Sevilla et al. (2022).** Also in the proceedings of IJCNN 2022, DOI 10.1109/IJCNN55064.2022.9891914 (arXiv metadata).
11. **Pages 1.6 and 4.4, DeepSeek-AI (2025), DeepSeek-R1.** Also in *Nature* 645: 633-638 (2025), DOI 10.1038/s41586-025-09422-z (arXiv metadata).
12. **Page 8.6, note 3, Kwa et al. (2025).** arXiv lists it as NeurIPS 2025.
13. **Page 1.3, note 5, He, Zhang, Ren, and Sun (2015), "Delving deep into rectifiers."** The citation gives a short title and no venue. The link goes to the ICCV 2015 paper, DOI 10.1109/ICCV.2015.123 (full title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification").

## Links that point to a different edition or copy than the one cited

14. **Page 1.2, note 4, Minsky and Papert (1969), *Perceptrons*.** DOI 10.7551/mitpress/11301.001.0001 is the MIT Press 2017 reissue of the expanded edition, not the 1969 first edition.
15. **Page 7.1, note 3, Jeffrey (1965), *The Logic of Decision*, McGraw-Hill.** The link goes to the University of Chicago Press page for the second edition, which is the edition in print. The 1965 edition has no page online.
16. **Page 1.2, note 5, Lighthill (1973).** The link goes to the full text of the report hosted by the Chilton Computing history site (Rutherford Appleton Laboratory), not to a publisher.
17. **Page 1.3, note 4, Hochreiter (1991).** The link goes to the scan of the thesis on Jürgen Schmidhuber's IDSIA page (title page checked).
18. **Page 7.1, note 4 and page 7.3, note 3, Ng and Russell (2000).** No DOI exists. The link goes to the PDF on Andrew Ng's Stanford page (first page checked).
19. **Page 8.4, note 1, Omohundro (2008).** No DOI was found for the IOS Press volume. The link goes to the author's PDF at selfawaresystems.com (first page checked).
20. **Pages 1.5, 3.1, and 3.7, Radford et al. (2019).** The link goes to the PDF on OpenAI's CDN (first page checked). The paper has no other official page.

## Links whose target could not be opened from this environment

21. **Page 1.6, note 2, OpenAI (2022), "Introducing ChatGPT."** `openai.com/index/chatgpt/` answers 403 to scripts. A web search for a sentence from the post returns this address, but the title, the date, and the quoted sentence about InstructGPT could not be read. The other four OpenAI posts (pages 1.6, 4.1, 8.2) were opened through a fetch tool, and their titles and dates match the citations; page 4.1's section name "Continuing to scale reinforcement learning" was confirmed.
22. **Page 1.4, note 1, ILSVRC 2012 results.** `image-net.org` does not answer through this environment's proxy, so `tools/check-links.mjs` reports it as broken here. A fetch tool opened the page: it is titled "ImageNet Large Scale Visual Recognition Competition 2012 (ILSVRC2012)" and lists SuperVision first with top-5 error 0.15315.
23. **Page 6.3, note 3, Project Gutenberg eBook 1342.** gutenberg.org answered 503 ("Temporarily Down for Maintenance") during the audit. A web search confirms the address is *Pride and Prejudice* by Jane Austen.

## References left without a link

- **Page 1.2, note 3, Novikoff (1962), "On convergence proofs on perceptrons."** No DOI and no stable official copy found.
- **Page 3.1, note 2, Gage (1994), "A New Algorithm for Data Compression," *C Users Journal*.** No DOI. Only unofficial copies exist online.
- **Page 8.1, note 1, Goodhart (1975), "Problems of monetary management: the U.K. experience," Reserve Bank of Australia.** The 1975 volume is not online. Crossref has the 1984 reprint, a chapter in Goodhart, *Monetary Theory and Practice* (DOI 10.1007/978-1-349-17295-5_4), which is a different publication from the one cited.
- **Page 8.4, note 2, Bostrom (2014), *Superintelligence*.** The book has no DOI, and the Oxford University Press page could not be opened to check it. The 2012 article in the same note is linked.
- **Page 1.4, note 5, "Russakovsky et al. (2015)" and page 2.7, note 1, "Griewank and Walther (2008)".** Mentions without a title. Russakovsky et al. is linked in note 1 of page 1.4.
- **Software and data credits** (js-tiktoken on page 3.1, the BVLC AlexNet model on page 1.4): not bibliographic references.

## Outside the scope of this audit

- Page and table numbers inside sources ("p. 308", "Table 3", "Theorems 5 and 6", "appendix C.2.1", "Figure 2") and what the notes say the sources show were not checked.
- At 375 px, with every sidenote opened, pages 1.3, 2.8, 5.2, and 5.3 scroll sideways by 135 px. This is the same on the commit before this branch and does not involve the links. `tools/check.mjs` does not open sidenotes, so it reports these pages ok. Page 6.2 had the same problem (67 px) from a long file path in a sidenote; the `overflow-wrap` rule added to `site.css` fixes it.
