# Pretraining Data Sources and Composition

## 1. Framing the Problem

Every pretraining run starts with a decision that gets far less public attention than architecture or optimizer choice, yet arguably has a larger effect on the resulting model's behavior: what text goes in, and in what proportions. Two models with identical parameter counts, identical attention mechanisms, and identical optimizer hyperparameters can behave completely differently if one was trained on a corpus that is 80% unfiltered web crawl and the other on a corpus with heavy curated-source upweighting.

Data composition is not a preprocessing footnote; it is a first-class modeling decision with the same order of consequence as choosing model depth versus width, and it is one of the areas where the gap between "reading a paper" and "having actually built a training corpus" is largest, because papers routinely compress months of data-engineering work into a single table or even a single sentence.

This file is about the *sources* that make up pretraining corpora and how labs combine them — not about the downstream cleaning mechanics (deduplication algorithms, quality classifiers, perplexity filtering), which belong in a companion file on filtering and deduplication, and not about the legal and licensing dimension of where this text is allowed to come from, which belongs in `008_Data_Licensing_Copyright_And_Governance.md`. It is also distinct from, though related to, the question of how a tokenizer's vocabulary is built over this data, covered in `004_Tokenizer_And_Vocabulary_Construction_At_Scale.md`.

The concern here is narrower and more concrete: where does the raw text come from, why does almost every serious pretraining corpus end up being a *mixture* of qualitatively different sources rather than a single source scaled up, and what do the actual disclosed mixtures from GPT-3, Llama 1, Llama 3, DeepSeek-V3, and Qwen2.5 look like in practice.

## 2. Common Crawl and Web-Scale Text

### 2.1 What Common Crawl mechanically is

Common Crawl is a nonprofit organization that has run a large-scale, roughly-monthly crawl of the public web since 2008, publishing the results as a freely downloadable dataset hosted on Amazon S3. It is not a curated dataset — it is closer to "what a search-engine-scale web crawler saw."

Its outputs are the raw material almost every LLM pretraining pipeline built in the last several years has ingested in some form, either directly or through an intermediate derivative (C4, CCNet, RefinedWeb, FineWeb, and similar pipelines are all, at bottom, reprocessed Common Crawl).

A single monthly Common Crawl snapshot is released in three parallel file formats, and understanding what each one actually contains matters for understanding why the "raw" numbers you see quoted (e.g., "Common Crawl has trillions of tokens") are almost never the number that ends up in a training corpus:

- **WARC (Web ARChive)** files are the rawest artifact: the full captured HTTP request and response for each crawled URL, including HTTP headers, and the response body exactly as served — typically raw HTML, but also images, PDFs, and other content types the crawler happened to fetch. WARC is the format search engines and archival projects (the Internet Archive uses the same format) have standardized on for exactly this reason: it preserves the transaction, not just an interpretation of it.
- **WAT (Web Archive Transformation)** files contain metadata *extracted* from the WARC records — outbound links, HTTP headers, content-type, title tags — as JSON, without the full page body. WAT is mostly useful for link-graph analysis and metadata-driven filtering, not text extraction per se.
- **WET (WARC Encoded Text)** files contain Common Crawl's own best-effort plaintext extraction from each WARC record: HTML tags stripped, leaving what is meant to be "the text of the page." This is the artifact most naively associated with "the text version of Common Crawl," and it is the starting point for many pipelines, though a number of serious pretraining efforts (Gopher/MassiveText, LLaMA's CCNet-based pipeline, RefinedWeb) do their own boilerplate and text extraction directly from WARC rather than trusting Common Crawl's default WET extraction, because that default extraction is fairly crude — it does not, for instance, reliably distinguish navigation menus, cookie-consent banners, and footer boilerplate from actual article body text.

A single monthly snapshot is enormous by any pre-LLM-era standard and covers on the order of two to three billion web pages, with compressed WARC size commonly in the tens of terabytes per month (the exact figure fluctuates snapshot to snapshot and has grown over the crawl's history) and the plaintext WET representation considerably smaller — commonly cited at a compressed size in the low tens of terabytes as well, translating into a token count that is genuinely enormous, plausibly hundreds of billions to low trillions of raw tokens *per month* before any filtering is applied, and Common Crawl's cumulative archive spans well over a hundred monthly snapshots at this point.

Treat the precise per-month figures as approximate and time-varying rather than a fixed constant — Common Crawl does not publish a single canonical "tokens per snapshot" number, and different processing pipelines report different post-extraction totals for what is nominally the same underlying crawl, because the extraction and quality-filtering choices materially change the surviving token count.

### 2.2 Why raw Common Crawl is extremely noisy

The critical thing to internalize, if you have never actually opened a WET file, is how bad the raw signal-to-noise ratio is. A crawler does not distinguish a well-written encyclopedia article from a spam page auto-generated to game search-engine rankings, a legal boilerplate page, a cookie-consent notice rendered as page text, a product listing page that is 90% navigation chrome and 10% actual content, or a duplicate of a page that exists verbatim on thousands of mirror sites.

Concretely, the noise in raw Common Crawl breaks down into several recurring categories that every pretraining pipeline has to contend with:

- **Boilerplate and template noise**: navigation menus, headers, footers, "Subscribe to our newsletter" prompts, cookie banners, and other chrome that is structurally part of the HTML but not part of the content a human reader would consider "the text of the page." A naive text extractor pulls all of this in as if it were prose.
- **Non-prose content**: pages that are mostly tables, structured data dumps, auto-generated listings (real estate listings, product catalogs, SEO-farmed keyword pages), or machine-translated text that is grammatically well-formed but semantically degenerate.
- **Spam and low-quality content farms**: pages specifically created to manipulate search rankings, often auto-generated or lightly reworded ("spun") from other content, which pollute a corpus with fluent-looking but low-information text.
- **Massive duplication**: the same content — news articles syndicated across outlets, legal disclaimers, terms-of-service boilerplate, quote/proverb pages — appears verbatim or near-verbatim across enormous numbers of distinct URLs. Left undeduplicated, this both wastes training compute on redundant signal and, more insidiously, can cause the model to overfit to memorizing frequently repeated strings, which is one of the empirically observed causes of unwanted verbatim memorization in LLMs.
- **Toxic and harmful content**: because Common Crawl indiscriminately captures the open web, it also captures the worst of the open web — hate speech, harassment, explicit content — none of which a lab typically wants represented at its natural web-frequency in a pretraining corpus without at least some filtering decision being made deliberately.

The practical consequence is that no lab trains directly on raw WET output. Every pretraining pipeline built on Common Crawl inserts a filtering stage between "raw crawl" and "training corpus" — quality classifiers (often trained to distinguish text similar to some reference corpus, such as Wikipedia-linked pages, from generic crawl text), language identification and filtering, n-gram or perplexity-based heuristic filters, and near-duplicate detection at scale (typically MinHash/LSH-style fuzzy deduplication, since exact-duplicate hashing catches only verbatim copies, not near-identical pages with a changed timestamp or ad slug).

This filtering stage is itself a major engineering undertaking and is covered in depth in the deduplication/filtering companion file rather than here — the point for this file is simply that "Common Crawl" as a training-data source name always implicitly means "some lab's filtered subset of Common Crawl," never the raw crawl itself, and different labs' filtering choices produce meaningfully different corpora from the same underlying raw material.

### 2.3 Why it remains the backbone anyway

Given how noisy it is, it is worth being explicit about why Common Crawl — or a heavily processed derivative of it — is nonetheless present in essentially every major pretraining corpus (GPT-3, Llama 1/2/3, DeepSeek, Qwen, and effectively every other frontier or near-frontier LLM trained since 2019). The answer is simply that no curated alternative comes remotely close to matching its raw scale, and raw scale is the resource that trillion-token-plus training budgets are fundamentally bottlenecked on.

Section 4 makes this quantitative, but the qualitative point is: Wikipedia, digitized books, arXiv, and Q&A forums combined do not add up to more than a small fraction of the tokens a modern pretraining run consumes. Common Crawl (and its derivatives, C4 and its many recent successors like RefinedWeb and FineWeb) is the only source whose raw scale is measured in the trillions of tokens rather than the billions, which makes it the only plausible way to fill a multi-trillion-token training budget at all — even after aggressive quality filtering discards the majority of raw crawl volume, what survives filtering is still far larger than every curated source combined.

Every mixture walked through in Section 5 reflects this: Common Crawl (in some filtered/processed form) is either the single largest component or ties for it in every disclosed mixture this file covers.

## 3. Curated Corpora

Curated corpora are the sources chosen specifically because they are dense in signal relative to their size — a token from Wikipedia or a peer-reviewed arXiv paper carries, on average, far more well-formed, factual, and stylistically clean information than a token from a random Common Crawl page.

They cannot replace web text at scale (Section 4), but nearly every disclosed pretraining mixture upweights them relative to their raw token share, on the premise that quality-per-token and diversity-of-register both matter independently of raw scale.

### 3.1 Books: two very different provenance stories

"Books" as a pretraining source actually names two structurally different kinds of corpora, and the distinction matters both practically and legally.

**Project Gutenberg** is a volunteer-run digitization effort of books whose U.S. copyright has expired — meaning the underlying works are in the public domain. A Gutenberg-derived corpus is unambiguous from a licensing standpoint: the text is legally free to redistribute and use for any purpose, including commercial model training, precisely because copyright protection has lapsed.

The tradeoff is that public-domain status skews the corpus toward older works (anything published, roughly, before the early 20th century in most jurisdictions, subject to the specific and sometimes complicated rules governing exactly which works have lapsed), so a Gutenberg-only book corpus systematically underrepresents contemporary vocabulary, contemporary topics, and modern prose style.

**Books3 and similar shadow-library-sourced corpora** are a different story entirely. Books3 (assembled as part of EleutherAI's "The Pile" corpus) and comparable large book corpora used by other labs are built from bulk collections of digitized books — including large numbers of works that are still under active copyright — sourced from shadow libraries (large-scale book-piracy repositories) rather than from licensed or public-domain channels.

This is precisely the category of data source that has become the center of the highest-profile copyright litigation against LLM developers: using Books3 (or corpora like it) means the training corpus contains full copies of copyrighted, commercially available books obtained without a license from the rightsholder.

The legal distinction from Gutenberg is not subtle — one is unambiguously permissible to use, the other is a live and materially contested legal question whose resolution differs across jurisdictions and is still being litigated as of this writing. This file is not the place to adjudicate that question or track ongoing litigation; that discussion, along with the general landscape of data licensing and copyright exposure across all source types (not just books), belongs in `008_Data_Licensing_Copyright_And_Governance.md`.

What matters here is simply naming the distinction accurately: when Section 5 discusses Llama 1's book corpus, it is disclosed by Meta's own paper as combining Gutenberg and Books3, and that combination is exactly this two-provenance mixture — one unimpeachable source and one legally contested one, sitting in the same "Books" bucket of the mixture table.

### 3.2 Wikipedia: disproportionate value for its size

Wikipedia is minuscule relative to Common Crawl in raw token count — English Wikipedia's full text, after stripping wiki markup, templates, and citation syntax, is commonly estimated at somewhere in the low single-digit billions of tokens (GPT-3's own Table 2.2 lists it at roughly 3 billion tokens for its English-only slice), which is a rounding error against a multi-trillion-token web corpus. Yet essentially every disclosed pretraining mixture upweights it well beyond its natural share, and it is worth being precise about why.

First, **information density**: Wikipedia articles are, on a per-token basis, close to maximally information-dense encyclopedic prose — there is very little filler, boilerplate, or redundancy compared to typical web text, so each token trains the model on comparatively more distinct factual and compositional signal.

Second, **factual reliability**, at least relative to the open web at large: Wikipedia's editorial process (imperfect, but real — citation requirements, edit history, community moderation) produces text that is, on average, meaningfully more reliable than an arbitrary web page, which matters when a model's factual grounding is a training input, not merely an output to be judged after the fact.

Third, **register**: Wikipedia's specific encyclopedic prose style — neutral point of view, structured exposition, consistent formatting conventions across a huge range of topics — is a register that barely exists elsewhere in web-scale text at this consistency and topical breadth, and having it well-represented plausibly helps the model learn a clean "explain this topic factually and concisely" mode that is useful both directly and as a component the model can blend with other registers at inference time.

The result is that Wikipedia is a case where "small but valuable" is not a hand-wavy claim — every lab that discloses per-source upweighting explicitly upweights it, and none treats it as merely "one more web source."

### 3.3 Code repositories

GitHub-derived corpora are now a standard mixture component, motivated by two distinct goals that get conflated in casual discussion but are worth separating. The first is training the model to write and reason about code directly — an increasingly commercially important capability in its own right.

The second, less obvious but well-supported by empirical results across multiple labs' ablations, is that code data appears to improve a model's general reasoning capability even on non-code tasks, plausibly because code enforces a uniquely strict, unambiguous, long-range-dependent structure (variable definitions must precede use, function signatures must match call sites, indentation and braces must nest correctly) that may transfer some benefit to the model's general capacity for structured, multi-step reasoning — though the precise mechanism behind this transfer is not fully settled science and should be presented as a widely observed empirical pattern rather than a proven causal mechanism.

Building a GitHub corpus responsibly requires license filtering as a first-class step, not an afterthought: GitHub repositories carry a wide range of licenses, from fully permissive (MIT, BSD, Apache 2.0) through copyleft (GPL family, which imposes redistribution obligations that are awkward to reconcile with training a model whose weights and outputs are distributed under different terms) to no explicit license at all.

A large fraction of public repositories have no LICENSE file at all, which under most jurisdictions' default copyright rules means the code is not actually permissively licensed just because it is publicly viewable on GitHub.

Llama 1's paper, for instance, explicitly discloses filtering its GitHub source to repositories distributed under permissive licenses (Apache, BSD, MIT) rather than taking all public repositories indiscriminately — a deliberate, disclosed licensing-aware filtering choice, in contrast to how casually some other web-text sources get treated.

The full landscape of license-filtering practice and its legal stakes is, again, the deeper subject of the data-licensing companion file; the point here is narrower: code corpora are exactly where "public" and "permissively licensed" visibly diverge, and every lab constructing a code-heavy mixture has to make an explicit filtering decision about which repositories qualify.

### 3.4 arXiv: scientific register at the cost of markup noise

arXiv is the preprint repository for physics, mathematics, computer science, and related quantitative fields, and its full-text archive is a source every major mixture discussed in Section 5 includes in some form. Its value proposition is a register almost absent elsewhere at scale: dense technical and mathematical prose, formal notation, and rigorous argument structure, none of which resembles typical web text or even typical books.

The mechanical complication is that arXiv papers are submitted predominantly as LaTeX source, and raw LaTeX source is full of markup that is not prose — macro definitions, bibliography and citation commands, figure/table environments, equation markup — that a naive ingestion pipeline would otherwise feed to the model as if it were natural text. Every disclosed arXiv-inclusive pipeline (Llama 1's paper explicitly calls this out) strips this LaTeX scaffolding down to something closer to the intended readable text plus mathematical notation, discarding macros and bibliography formatting that carry no linguistic signal.

This is a clean, small-scale illustration of a pattern that recurs across every curated source: raw source format and "the text you actually want to train on" are never quite the same thing, and building the extraction pipeline that bridges the two is nontrivial, source-specific engineering work even for a source as comparatively clean as arXiv.

### 3.5 StackExchange and forum data: a distinct structural register

StackExchange (the network of Q&A sites including Stack Overflow) contributes a register that is structurally unlike continuous prose: a question, followed by one or more candidate answers, typically rankable by community vote score, often interleaved with code snippets and terse, direct explanatory style very different from either encyclopedic Wikipedia prose or narrative book text.

Llama 1's paper discloses sourcing from the 28 largest StackExchange sites, sorted by answer score (a natural quality proxy — higher-voted answers are, on average, both more correct and more clearly written), with HTML markup stripped.

The reason this is worth calling a genuinely distinct register, not just "more web text," is the question-answer *structure* itself.

A model exposed to a substantial volume of Q&A-formatted data learns something about the discourse pattern of "here is a problem statement, here is a direct answer to it" that is comparatively rare in prose-heavy sources like books or news articles.

This pattern plausibly transfers directly to how well the base model responds when later prompted, post-training, in an instruction-following or chat format — the base model has already seen a large amount of naturally occurring "question, then answer" text before any SFT stage ever touches it.

## 4. The Web-Text-Versus-Curated-Text Trade-off, Made Concrete

Sections 2 and 3 already gesture at the trade-off; this section makes it quantitative enough to be useful in an interview setting, where a vague "web text is noisy but curated text is small" answer is a much weaker response than one backed by an actual sense of scale.

The core numbers, treated as approximate, order-of-magnitude figures rather than precise constants (exact counts vary by processing pipeline, wiki-dump date, and tokenizer):

- **English Wikipedia**, fully de-markup'd, is on the order of a few billion tokens — GPT-3's own disclosed figure is roughly 3 billion tokens for its Wikipedia slice, and this is broadly consistent with independent estimates of English Wikipedia's plain-text size run through a typical BPE-style tokenizer.
- **arXiv's full-text archive**, similarly, lands in the range of a few tens of billions of tokens once LaTeX markup is stripped — commonly cited open reproductions of arXiv-derived training subsets (for example, in the RedPajama and similar open replications of the Llama 1 mixture) report figures in roughly this range.
- **StackExchange**, likewise, is a source measured in the tens of billions of tokens at most across even its largest constituent sites, not hundreds of billions.
- **Digitized book corpora** are larger than any of the above individually — GPT-3's Books1 and Books2 slices are disclosed at roughly 12 billion and 55 billion tokens respectively — but still nowhere close to web-crawl scale, and expanding a book corpus further runs into the hard ceiling of how many distinct books actually exist and are digitized and accessible at all, a ceiling that does not move nearly as fast as compute budgets do.
- **Common Crawl**, by contrast, is measured in the hundreds of billions to low trillions of tokens *per monthly snapshot* even after aggressive filtering, with well over a decade of monthly snapshots available to draw from — meaning the addressable scale, even filtered, is easily multiple trillions of tokens, and arguably tens of trillions if a lab is willing to draw on many snapshots and tolerate the deduplication cost of doing so.

Put these numbers side by side and the design tension in Section 1 becomes a hard scale ceiling, not a soft stylistic preference. Sum every curated source discussed in Section 3 — Wikipedia, arXiv, StackExchange, a generous book corpus — and, even being charitable about book-corpus size, the total lands somewhere in the neighborhood of a few hundred billion tokens at the very most, and realistically closer to one hundred billion or less if book-corpus size is bounded to what is actually legally and practically obtainable.

Modern frontier pretraining runs consume multiple *trillions* of tokens — Llama 1 already used roughly 1.0–1.4 trillion tokens in 2023, Llama 3's initial release used over 15 trillion, and DeepSeek-V3 used 14.8 trillion.

A training run at that scale cannot be filled by curated sources alone even if a lab were willing to repeat every curated token across dozens of epochs. Repeating a fixed small corpus that many times produces measurably worse outcomes than single- or low-multi-epoch exposure to a much larger, more diverse corpus — this is not just an intuition; it is the empirical subject of data-constrained scaling-law work (e.g., Muennighoff et al., "Scaling Data-Constrained Language Models," 2023), which finds that returns from repeating a fixed corpus decay noticeably beyond roughly four epochs, and that beyond a moderate repetition count a lab is better off, compute-for-compute, mixing in even fairly noisy additional unique data than repeating the clean data further. "Just repeat the good stuff many times" is accordingly not treated as a viable substitute for genuine scale by any lab whose mixture is discussed in this file.

This is why every mixture in Section 5, without exception, is majority web text by raw token count, with curated sources present specifically as an upweighted minority rather than as the substrate itself. It is also why web text is not merely a scale-filler being tolerated grudgingly: it is the only source that contains huge swaths of register and topical diversity that curated corpora structurally cannot contain at all — casual conversational writing, product reviews, forum arguments, informal multilingual text, slang, opinion pieces, and the sheer topical breadth of "everything anyone has ever put on the public web," none of which Wikipedia's encyclopedic register, arXiv's technical register, or even a large book corpus's narrative register will ever cover.

The trade-off, stated precisely, is not "noisy-and-big versus clean-and-small as competing options where one should simply be preferred" — it is that the two source types are complementary along axes (scale and diversity-of-register on one side, density and reliability on the other) that no single source type covers alone, which is exactly why the empirical answer, across every lab covered in Section 5, has converged on "use both, and tune the mixing ratio," rather than either extreme.

## 5. Concrete, Disclosed Mixtures From Real Models

### 5.1 GPT-3's Table 2.2 mixture

GPT-3's paper (Brown et al., 2020) discloses one of the earliest widely cited mixture tables in the modern LLM literature — full detail and architectural context in `..\..\GPT\003_GPT3.md`.

Five named sources: Common Crawl (filtered) at roughly 410 billion available tokens, contributing roughly 60% of training-mix sampling weight; WebText2 at roughly 19 billion tokens contributing roughly 22%; Books1 at roughly 12 billion tokens contributing roughly 8%; Books2 at roughly 55 billion tokens also contributing roughly 8%; and Wikipedia at roughly 3 billion tokens contributing roughly 3%. The total training budget was approximately 300 billion tokens.

The number worth sitting with is not any individual percentage but the *mismatch* between raw availability and sampling weight. Common Crawl supplies by far the largest pool of available tokens (410B, before even counting how much more exists beyond what was filtered in) yet is *downweighted* to roughly 60% of a 300B-token budget — meaning it is heavily undersampled relative to its raw share, seen for well under one full pass through even the filtered pool.

Meanwhile Wikipedia, Books1, Books2, and WebText2 combined supply only about 89 billion raw tokens yet are collectively upweighted to roughly 40% of the training mix — meaning these smaller sources are seen multiple times over the course of training (several epochs' worth of repetition), on the explicit premise that curated, high-quality tokens are worth more per unit than an equivalent count of filtered web tokens.

This is the general shape every subsequent disclosed mixture in this file follows in spirit, even where the exact ratios differ: sampling weight tracks a lab's belief about per-token value, not raw token count, and constructing a pretraining corpus is therefore an active curation decision about repetition rates per source, not a passive union of everything available.

### 5.2 Llama 1's public-data-only, seven-source mixture

Llama 1 (Touvron et al., 2023; full detail in `..\..\OpenSource\001_Llama1.md`) discloses a considerably more granular seven-source mixture than GPT-3's five-source table, and does so specifically because Meta constrained itself to training exclusively on data it characterized as "publicly available and compatible with open sourcing" — a deliberate governance decision, not an incidental one, that is the direct predecessor to the broader data-licensing discussion in `008_Data_Licensing_Copyright_And_Governance.md` (which covers the licensing landscape across sources and labs in depth; this file does not duplicate that discussion). The disclosed mixture, by sampling proportion:

| Source | Sampling proportion |
|---|---|
| CommonCrawl | 67% |
| C4 | 15% |
| GitHub | 4.5% |
| Wikipedia (20 languages) | 4.5% |
| Books (Gutenberg + Books3) | 4.5% |
| ArXiv | 2.5% |
| StackExchange | 2% |

This trained the 7B and 13B models on roughly 1.0 trillion tokens and the 33B and 65B models on roughly 1.4 trillion tokens, with Wikipedia and Books upweighted to roughly two epochs relative to the other sources' roughly single-pass exposure — the same "upweight the dense, reliable sources" logic as GPT-3's mixture, expressed here as an explicit per-source epoch count rather than an implicit sampling-weight mismatch.

Two details are worth flagging precisely because they are easy to elide.

First, C4 (Raffel et al.'s cleaned Common Crawl derivative) is included *in addition to* raw CommonCrawl, not as a substitute for it — Meta's stated reasoning is that C4's distinct heuristic cleaning pipeline provides filtering diversity that a single CC-filtering approach would not, i.e., two different noisy-web sources cleaned two different ways are more valuable together than either alone, a subtlety that is easy to miss if you assume "Common Crawl" and "C4" are redundant.

Second, the Books component — Gutenberg plus Books3 — is exactly the two-provenance mixture flagged in Section 3.1: one component (Gutenberg) is unambiguously public-domain, the other (Books3, sourced from a shadow-library-derived corpus assembled as part of EleutherAI's Pile) sits inside the "publicly available" framing Meta used for the paper's overall data-sourcing claim in a way that later scrutiny — again, the subject of the licensing companion file, not this one — treated as considerably more contestable than Gutenberg's.

### 5.3 Llama 3's larger, less granular mixture

Llama 3 (initial release April 2024, extended in the "Llama 3 Herd of Models" July 2024 report; full detail in `..\..\OpenSource\003_Llama3.md`) trained its 8B and 70B models on over 15 trillion tokens — more than a tenfold increase over Llama 1's largest token budget — and this jump in scale is the direct, mechanical consequence of the scale ceiling quantified in Section 4.

Llama 1's original seven sources, even generously scaled, simply do not contain 15 trillion tokens' worth of quality-filtered text: Wikipedia, arXiv, and StackExchange are hard-capped by Section 4's figures regardless of how a lab chooses to weight them, GitHub's permissively licensed subset is large but not inexhaustible, and Books3-style corpora do not grow meaningfully faster than the world's supply of digitized books does.

Reaching 15T+ tokens at usable quality therefore required both far deeper mining of the web-text pool (more Common Crawl snapshots, more aggressive but still quality-preserving filtering, and reportedly using earlier Llama models themselves as data-quality classifiers to sift a much larger raw pool than a simple heuristic filter could) and meaningfully broadening source diversity beyond the original seven-source list, with the paper describing materially increased emphasis on code, multilingual text, and reasoning/math-oriented data relative to Llama 2's mixture.

What is conspicuously different from Llama 1's disclosure is the granularity. Meta's Llama 3 paper does not publish a per-source percentage table analogous to Llama 1's seven-row table above — the composition is described qualitatively (more code, more multilingual data, more reasoning-oriented data than Llama 2, a "data annealing" phase that upweights a smaller, higher-quality mixture in the final stage of pretraining) rather than itemized as exact fractions per named source. This is worth stating plainly rather than glossing over: Meta disclosed substantially less granular composition detail for a model trained on ten times the tokens and representing a considerably larger research investment than Llama 1.

Whether this reflects the mixture becoming genuinely too complex and multi-stage to summarize in one table, a more restrictive external-disclosure posture as Llama became a more commercially important product line, or some combination of both is not something the paper itself addresses, and should be treated as an open question rather than resolved with unfounded speculation about motive.

### 5.4 DeepSeek-V3 and Qwen2.5: two different disclosure postures

DeepSeek-V3 (technical report, December 2024; full detail in `..\..\OpenSource\007_DeepSeek_V3.md`) trained on 14.8 trillion tokens, described by DeepSeek as a high-quality, multilingual (English- and Chinese-heavy) corpus with increased representation of math and code tokens relative to a typical web-dominated mixture.

That description is close to the full extent of what DeepSeek's report discloses about data composition — there is no per-source percentage table, no named list of constituent corpora analogous to Llama 1's, and no granular breakdown of what fraction of the 14.8T tokens is web-derived versus curated versus synthetic.

This is consistent with a broader pattern already visible in DeepSeek-V3's disclosure posture elsewhere (Section 3 of the DeepSeek-V3 file notes the same granularity gap for data as exists for training infrastructure, relative to how detailed the report is about architecture and training cost) — DeepSeek is unusually transparent about compute cost and architectural mechanism, and comparatively opaque about the data pipeline itself. The honest statement here is simply that DeepSeek-V3's exact data-mixture percentages are not publicly disclosed, and any specific breakdown beyond "large, multilingual, math/code-enriched" would be invented, not reported.

Qwen2.5 (Alibaba, technical report December 2024; full detail in `..\..\OpenSource\009_Qwen2_5.md`) discloses somewhat more about its data strategy than DeepSeek-V3, without reaching Llama 1's level of itemization. The disclosed facts: a pretraining corpus of roughly 18 trillion tokens (up from Qwen2's 7 trillion), with Alibaba's report stating increased emphasis on math, code, and multilingual data relative to prior Qwen generations, plus a synthetic-data component generated by earlier Qwen models and filtered for quality — a "self-improvement" data-generation loop disclosed at a summary level rather than with exact synthetic-to-organic token ratios.

Multilingual coverage across more than 29 languages is stated as an explicit design goal, consistent with Alibaba's product need for strong Chinese-language performance in a way that Meta's English-centric Llama line does not share to the same degree.

Beyond these disclosed facts — the token count, the stated math/code/multilingual emphasis, and the synthetic-data loop — the report does not provide a named per-source percentage breakdown, and exact filtering methodology and synthetic-data ratios are not published in full detail; this file reports only what Alibaba's technical report actually states rather than inferring specifics it does not disclose.

The pattern across all four models in this section is itself an interview-relevant observation: disclosure granularity for pretraining data composition has, if anything, gone *down* over time even as token budgets have gone up by more than an order of magnitude, and the two labs disclosing the least detail here (DeepSeek and, to a lesser extent, Qwen) are also, not coincidentally, the two labs disclosing the most detail about training infrastructure and cost elsewhere in the same reports. Data composition appears to be treated by multiple labs as more sensitive or more genuinely difficult to summarize cleanly than compute infrastructure, and a careful answer should notice and name this asymmetry rather than assume every lab treats disclosure uniformly across topics.

To make this asymmetry concrete, the following table summarizes what is actually disclosed, side by side:

| Model | Disclosed token count | Named per-source breakdown | Multilingual disclosure |
|---|---|---|---|
| GPT-3 | ~300B (5 named sources) | Yes — Table 2.2, full weights | English-only sources named |
| Llama 1 | ~1.0–1.4T (7 named sources) | Yes — full percentage table | Wikipedia in 20 languages, no per-language token split |
| Llama 3 | 15T+ | No — qualitative only | "increased" multilingual data, no breakdown |
| DeepSeek-V3 | 14.8T | No — one-sentence description | English/Chinese-heavy, no breakdown |
| Qwen2.5 | ~18T | No — qualitative + token count only | 29+ languages claimed, no per-language split |

This table is a useful thing to have memorized in rough shape for an interview: it shows the field converging on far larger token budgets while, if anything, disclosing less about how those tokens are actually composed — the opposite direction from what one might naively expect as the field "matures."

## 6. Multilingual Composition as a Deliberate, Contested Allocation

Every one of the mixtures walked through in Section 5 embeds an implicit or explicit decision about how many pretraining tokens go to languages other than English, and this decision deserves to be named as a genuine, unavoidable design choice rather than something that falls out automatically from "just use the web." It does not fall out automatically, because raw web-text availability is itself wildly skewed toward English and a small number of other high-resource languages (Chinese, Spanish, French, German, and a handful of others), so even an unfiltered, unweighted crawl-based corpus is already far from linguistically balanced before a lab makes a single explicit weighting choice.

A lab that does nothing to correct for this default skew has, in effect, made the choice to be English-dominant by omission.

The trade-off runs in both directions, and neither side is free. A heavily English-dominant mix — closer to Llama 1's original design point, where Wikipedia's 20-language coverage was a comparatively small, upweighted slice inside an otherwise English-and-code-heavy mixture — buys better English (and, correlated with English web-text volume, better code) capability per unit of compute spent, precisely because high-quality English web text is the most abundant raw material available at any filtering quality bar.

The cost is a model that is measurably weaker in every other language, an increasingly hard-to-justify limitation for any lab targeting a genuinely global product surface, which is exactly the gap Qwen's Chinese-market-driven design goal and Llama 3's "much more multilingual than Llama 2" framing are both explicit, disclosed responses to.

Conversely, a more deliberately multilingual mix improves non-English capability, but this is not a free reallocation — it dilutes the effective token budget available for English (and, again correlated, code) at a fixed total training-token count.

It also runs headlong into a second problem that raw token-share numbers alone do not capture: high-quality web text is considerably scarcer and noisier for the overwhelming majority of the world's languages than it is for English. A lab cannot simply decide "allocate tokens proportional to global speaker population" and expect equal quality per allocated token across languages, because the underlying raw material to draw from is not equally abundant or equally clean to begin with — a token's worth of Icelandic web text and a token's worth of English web text are not, in practice, interchangeable in expected quality, however evenly a mixture recipe might try to weight them.

Equal token-count "fairness" across languages is therefore not the same thing as equal achieved-quality fairness across languages, and no lab covered in this file has published a mixture that claims otherwise.

This tension is closely related to, but analytically distinct from, the tokenizer fairness problem discussed in `004_Tokenizer_And_Vocabulary_Construction_At_Scale.md` — that file addresses how a fixed vocabulary's byte-pair or byte-level merges end up compressing some languages far more efficiently than others (meaning a fixed context window holds meaningfully less actual content for a poorly-tokenized language, and generation costs correspondingly more per unit of output text), which is a downstream *consequence* partly shaped by which languages a corpus overrepresents during tokenizer training.

The concern here is the upstream *pretraining-token-allocation* decision itself, independent of how efficiently the tokenizer subsequently encodes whatever text is allocated.

The two problems compound — a language underrepresented in the pretraining mixture is frequently also a language whose tokenizer compression is worse, because the same lack of abundant clean training text that makes it hard to allocate a large, high-quality pretraining share to a language also makes it hard to build a well-fitted, efficient tokenizer vocabulary for it — but they are solved by different levers (mixture weighting versus vocabulary construction) and should not be conflated when discussing either one specifically.

## 7. What a Staff-Level Interviewer Is Actually Listening For

An interviewer asking about pretraining data composition at the staff level is very rarely testing whether you can recite Llama 1's seven-source percentage table from memory — that is trivia, and reciting it correctly proves only that you read a paper carefully.

What is actually being probed is whether you understand data composition as a *trade-off management problem under a hard scale constraint*: can you explain, unprompted, why no lab trains on curated sources alone even though curated sources are unambiguously higher quality per token, and can you back that explanation with an actual sense of the relevant magnitudes (billions for Wikipedia and arXiv, trillions for a modern training budget) rather than a vague gesture at "there's not enough of it."

A strong answer also shows judgment about disclosure: noticing, as this file does in Section 5, that granular data-composition disclosure has gotten sparser even as token budgets have exploded, and being willing to say plainly when a specific number is genuinely undisclosed (DeepSeek-V3's exact mixture percentages) rather than inventing a plausible-sounding figure to fill the silence.

Finally, the multilingual allocation question is a good test of whether you can hold two true things in tension at once — that under-serving non-English languages is a real, deliberate cost imposed by an English-dominant mixture, and that "just allocate tokens proportionally" is not a solution because the underlying raw material is not equally abundant or clean across languages — without collapsing the answer into either "just add more languages, problem solved" or "the imbalance is unavoidable, nothing to discuss." That capacity to sit with a genuine trade-off rather than resolve it into a slogan is precisely the signal a staff-level conversation on this topic is designed to surface.
