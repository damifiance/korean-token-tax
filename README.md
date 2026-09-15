# Why does Korean pay more tokens? (한국어의 "인코딩 세금" 검증)

An information-theoretic study of Korean–English tokenization using parallel
texts, a fixed language model, and decision rules frozen before held-out
scoring. It measures **model-relative code bits per billing token**, not
intrinsic language entropy or the usefulness of an LLM's answer.

The held-out Korean texts used **38.9% more tokens** and received **31.2%
more Qwen-relative code bits**. Their bits-per-token ratio was **5.5% lower**,
but the prespecified minimum-effect criterion was not met. The result is
retained; the study does not establish why Korean received more model bits.

[Read the paper (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex) ·
[Frozen specification](pipeline/spec/measurement_spec.yaml) ·
[Archived results](pipeline/results/confirmatory/analysis_summary.json)

## 1. Research question

When aligned Korean and English messages receive different token counts,
how much of the difference is accompanied by a difference in model-assigned
code length, and how sensitive is it to the billing tokenizer?

The original motivation contrasted an **information explanation** with an
**encoding explanation**. The expected encoding-only pattern was a large
token ratio but approximately equal total bits. However, these are not
mutually exclusive causal mechanisms, and the measurements do not cleanly
separate them. Equivalent meaning does not imply equal Shannon entropy;
poor model prediction can also increase measured bits.

The experiment therefore tests a specified efficiency contrast and tokenizer
sensitivity. It does **not** identify an intrinsic information difference
between the two languages or isolate tokenizer training-language mix as the
cause of the token gap.

## 2. What is measured?

### Two separate measurements of the same text

For each text `s`, the billing tokenizer counts content tokens, `T_b(s)`.
Separately, Qwen uses **its own tokenizer** and next-token probabilities to
compute a model-relative sequential code length:

```text
z = τ_M(s)
B_M(s) = −Σ_j log₂ Q_M(z_j | c, z_<j)
```

Here `M` is the fixed Qwen estimator, `z_j` is a Qwen token, and `c` is the
document-boundary context. Every content token is scored; EOS is not. The
billing tokenizer's token IDs are used only for counting and are never fed
into Qwen.

`B_M` is an ideal code length in bits. Arithmetic coding can approach this
sequence-level length with finite coding overhead and suitable framing;
this study computes log probabilities, not actual compressed files.
The value is independent of the **billing tokenizer**, but still depends on
Qwen, its own tokenizer, and the scoring policy.

### Corpus totals and ratios

For language `L` (Korean or English), sum over the same included pairs:

```text
B_L = Σ_i B_M(s_i,L)
T_L = Σ_i T_b(s_i,L)
η_L = B_L / T_L
```

Thus `M` identifies the estimator, while `L` identifies the language being
aggregated. The study's operational efficiency measure `η_L` has units of
**Qwen-relative code bits per billing token**.

| Quantity | Definition | Interpretation |
|---|---|---|
| `R_T` | `T_KR / T_EN` | Baseline token ratio |
| `R_B` | `B_KR / B_EN` | Qwen-relative code-length ratio |
| `R_η` | `η_KR / η_EN = R_B / R_T` | Relative bits-per-token efficiency |
| `1 − R_η` | Relative efficiency shortfall | Not lost information or a measured fraction of wasted spending |
| `C_K` | `T_KR,polyglot / T_KR,o200k` | Korean token-count sensitivity on identical strings |
| `G` (exploratory) | `R_T,polyglot / R_T,o200k` | Change in the language token ratio; can also fall if English token counts rise |

These are **ratios of pooled totals**, not means of per-message ratios.
`R_T`, `R_B`, and `R_η` are algebraically dependent, not independent findings.

### Why model bits are not intrinsic entropy

For a source distribution `P` and model distribution `Q`, expected
cross-entropy satisfies `H(P,Q) = H(P) + D_KL(P‖Q)`. Measured code length
therefore combines source uncertainty with model mismatch. The Korean and
English mismatch terms are unknown; this study does not establish that the
Korean term is larger or quantify its effect on the ratio. Translation
choices, genre, and possible training-data contamination add further limits.

## 3. Prospective specification freeze

The corpus was split into a **pilot**, used to check implementation and
estimate precision, and a **held-out set** for the final comparisons.
Thresholds were chosen after examining the pilot but before scoring the
held-out set. They are study-specific substantive margins, not universal
information-theoretic constants.

The specification was frozen on **10 September 2026 at 07:14 UTC**. Its
SHA-256 records the agreed configuration; confirmatory analysis rejects an
unfrozen spec, a content-hash mismatch, or pilot-tagged input. This is an
internal prospective freeze, **not an externally registered preregistration**.
A hash is not an independent timestamp or proof that data remained unseen.
The original spec and results are preserved in the repository.

Frozen decisions ([measurement_spec.yaml](pipeline/spec/measurement_spec.yaml),
hash prefix `850fa9be…`):

| Decision | Value | Rule applied to the held-out bootstrap |
|---|---|---|
| Primary endpoint | `R_η` | one-sided 95 % upper bound < `c_η = 0.95` |
| Token ratio | `R_T` | one-sided 95 % lower bound > `c_T = 1.20` |
| Tokenizer control | `C_K` | one-sided 95 % upper bound < `c_C = 0.90` |
| Overall | all three must pass | `all_confirmatory_pass` |
| Bit-ratio equivalence (exploratory) | `R_B` | 90 % CI inside `[0.8333, 1.20]` |
| Exclusions | drop identical Korean/English strings and rows with no Hangul on the Korean side | applied identically to pilot and held-out |

Length breakdowns, permutation tests, per-pair distributions, and `G` are
exploratory. Historical-tokenizer and split-sensitivity analyses were added
post hoc. FLORES+ is planned as exploratory replication only; its corpus has
not been downloaded or scored. Revising thresholds now would not create a
new confirmatory test on the already-inspected held-out data.

## 4. Setup

| Component | Choice | Pinned identity |
|---|---|---|
| Corpus | OPUS Global Voices v2018q4 en–ko | 9,017 aligned pairs in 347 nonempty articles; one news/citizen-media genre |
| Estimator model `M` | Qwen/Qwen3-1.7B-Base | revision `ea980cb0…`, bf16 weights, float32 log-softmax; original scoring on MPS |
| Baseline billing tokenizer `b` | OpenAI `o200k_base` (tiktoken 0.14.0) | vocabulary asset SHA-256 verified |
| Korean-trained comparison tokenizer | EleutherAI/polyglot-ko-1.3b | revision `557e162c…`; tokenizer only, not the Polyglot language model |
| Token accounting | content only (no chat wrapper) | |
| Scoring policy | prepend the model's document boundary token, score every content token, no EOS | |
| Statistics | ratio-of-sums estimands; paired cluster bootstrap (whole articles resampled, both languages together), 10,000 replicates | seed 20260910 |

The three corpus counts describe different levels: each row is a bilingual
pair, pairs belong to articles, and the articles share one broad genre.
The deterministic split (seed `20260910`) assigned 69 articles / 1,732 raw
pairs to the pilot and 278 articles / 7,285 raw pairs to the held-out set.
Articles connected by exact bilingual duplicates were assigned together;
neither an article nor an exact bilingual duplicate crossed the split.

Text processing was limited to Unicode NFC normalization and trimming
surrounding whitespace, without rewriting. NFC standardizes canonically
equivalent Unicode sequences; it is not transliteration. Translation
direction and Qwen training-data contamination remain unknown.

Whole articles are resampled with both languages together because sentences
within an article are dependent. The identity `R_η = R_B / R_T` is preserved
in every bootstrap replicate. Full model revisions and scoring settings are
recorded in the frozen spec and archived provenance.

Content-only counts are a **billing proxy**, not reconstructed API invoices:
chat wrappers, output tokens, caching, and differing price schedules are not
measured. The tokenizer comparison does not imply that an existing model
can use the other tokenizer without adaptation.

## 5. Results

<!-- RESULTS:BEGIN -->
Held-out set after exclusions: **7,046 pairs in 278 articles** (239 rows dropped: drop_ko_equals_en 104, drop_ko_no_hangul 135). Pilot after the same rules: 1,672 pairs in 69 articles.

### Pooled estimands (held-out, 95 % cluster-bootstrap percentile CI)

| Quantity | Point | 95 % CI |
|---|---|---|
| `R_T` token ratio KR/EN (o200k) | 1.3888 | [1.3714, 1.4058] |
| `R_B` Qwen-relative code-length ratio KR/EN | 1.3121 | [1.2959, 1.3278] |
| `R_η` = `R_B` / `R_T` | 0.9448 | [0.9363, 0.9532] |
| `1 − R_η` model-relative efficiency shortfall | 0.0552 | [0.0468, 0.0637] |
| `C_K` Korean tokens, polyglot-ko / o200k | 0.8665 | [0.8601, 0.8729] |
| `R_T` under polyglot-ko (exploratory) | 0.5095 | [0.5025, 0.5163] |
| `G` change in language token ratio (exploratory) | 0.3668 | [0.3629, 0.3707] |

### Prospectively frozen decisions

| Comparison | Rule | Bound observed | Result |
|---|---|---|---|
| `R_eta` | one-sided 95% upper bound < c_eta=0.95 | 0.9519 | FAIL |
| `R_T` | one-sided 95% lower bound > c_T=1.2 | 1.3741 | PASS |
| `C_K` | one-sided 95% upper bound < c_C=0.9 | 0.8719 | PASS |

**Overall (all_confirmatory_pass): FAIL.**

The overall rule was not met because `R_eta` did not satisfy "one-sided 95% upper bound < c_eta=0.95" (bound 0.9519; point estimate 0.9448). The thresholds were frozen before this data was scored and are not revised here; the point estimates and intervals above are reported as observed.

Exploratory bit-ratio equivalence: 90 % CI for `R_B` = [1.2984, 1.3252] against [0.8333, 1.20] → not inside the interval. Qwen assigned Korean greater total code length. The frozen primary criterion required the one-sided 95% upper bound for `R_η` to be below 0.95; approximate equality of total code lengths was assessed only exploratorily.

### Pilot vs held-out

| Quantity | Pilot (69 articles) | Held-out (278 articles) |
|---|---|---|
| `R_T` | 1.3769 | 1.3888 |
| `R_B` | 1.2943 | 1.3121 |
| `R_η` | 0.9400 | 0.9448 |
| `C_K` | 0.8697 | 0.8665 |

Pilot values here use the frozen exclusions, not the full raw pilot sample.

### By message length (exploratory; terciles of `n_chars_en`)

| Stratum | Observed length range (chars) | Pairs | `R_T` | `R_B` | `R_η` [95 % CI] | `C_K` |
|---|---|---|---|---|---|---|
| Q1 | 3–74 | 2349 | 1.564 | 1.419 | 0.907 [0.894, 0.921] | 0.882 |
| Q2 | 74–137 | 2348 | 1.415 | 1.318 | 0.931 [0.919, 0.944] | 0.868 |
| Q3 | 137–587 | 2349 | 1.329 | 1.271 | 0.956 [0.946, 0.966] | 0.861 |

Groups use length ranks, with ties broken by row order; the displayed minimum–maximum ranges can overlap. These are descriptive subgroup estimates, not a direct test of differences between groups.

### Paired cluster permutation tests (exploratory; 10,000 sign-flips of whole articles)

| Quantity | log pooled ratio | two-sided p |
|---|---|---|
| `R_T` | +0.3284 | 0.00010 |
| `R_B` | +0.2717 | 0.00010 |
| `R_eta` | -0.0567 | 0.00010 |
| `C_K` | -0.1433 | 0.00010 |

Minimum attainable p is 1/(1+10,000). The three language ratios are algebraically dependent: `R_η` is determined by `R_B` and `R_T`. These exploratory tests assume language-label exchangeability (tokenizer-label exchangeability for `C_K`); they do not establish the prespecified effect magnitudes.

### Per-message ratio distribution (descriptive only; not the estimand)

| Ratio | mean | median | geometric mean | 5th pct | 95th pct |
|---|---|---|---|---|---|
| `R_T` | 1.489 | 1.429 | 1.402 | 0.762 | 2.364 |
| `R_B` | 1.396 | 1.357 | 1.331 | 0.766 | 2.091 |
| `R_eta` | 0.973 | 0.944 | 0.949 | 0.666 | 1.369 |

Archived analysis outputs, with source text omitted, include 10,000 bootstrap replicates, provenance (package versions, tokenizer file hashes, dataset hash, device) and the pilot precision simulation: see [pipeline/results/](pipeline/results/).
<!-- RESULTS:END -->

## 6. Interpretation and unresolved explanations

### Why the efficiency gap was smaller than expected

The numerator and denominator both increased:

```text
R_η = (B_KR / B_EN) / (T_KR / T_EN)
    = 1.3121 / 1.3888
    ≈ 0.9448
```

Korean's 31.2% greater model code length partly offsets its 38.9% greater
token count in `B/T`, leaving a 5.52% relative efficiency shortfall. This is
an arithmetic explanation of the point estimate, not a causal explanation
of why either component increased.

The formal failure is a separate issue: the one-sided 95% upper bound for
`R_η` was **0.9519**, not below **0.95**. The data indicate a lower ratio,
but do not establish a reduction exceeding the prespecified 5% margin at
that confidence level. The failed criterion does not make the ratio invalid
or prove that the two languages are equally efficient.

### Could Korean linguistic structure explain the larger numerator?

This is a **post-hoc hypothesis**, not a result. Korean morphology, particles,
and ending combinations could affect written-text uncertainty or the fixed
model's ability to predict it. But grammatical complexity does not by itself
imply higher Shannon entropy: entropy requires a source distribution and a
specified unit, and predictable structure can reduce conditional uncertainty.
Per-character entropy is also not the same quantity as total message code
length.

The present measurements cannot distinguish greater source uncertainty from
greater Qwen mismatch, translation effects, or genre effects. Consequently,
the observation that both `B` and `T` increased does **not** establish that
Korean intrinsically contains more information. The paper discusses this
possibility with reference to [Mielke et al. (2019)](https://aclanthology.org/P19-1491/).

### What does “value” mean here?

`B/T` is an operational coding-efficiency measure, not a validated measure
of semantic value, answer quality, or benefit to a user. A less accurate
estimator can assign more bits to unchanged text and thereby raise `B/T`
without adding content. Likewise, the 5.5% shortfall is **not 5.5% information
loss**, a count of empty tokens, or an established fraction of wasted money.
“Encoding tax” is a motivating label, not a demonstrated monetary loss.

### What the tokenizer comparisons establish

Polyglot-Ko used about **13.3% fewer Korean tokens** on the same strings
(`C_K = 0.8665`). This establishes tokenizer sensitivity. Because the two
tokenizers also differ in vocabulary size and other design choices, it does
not isolate English-heavy training as the cause or rule out source-language
differences.

The paper also reports a **post-hoc** comparison holding the held-out strings
and Qwen bits fixed while changing only the billing encoding:

| Billing encoding | `R_T` | `R_η` | Model-relative efficiency shortfall |
|---|---|---|---|
| `r50k_base` | 4.591 | 0.286 | 71.4% |
| `cl100k_base` | 2.238 | 0.586 | 41.4% |
| `o200k_base` | 1.389 | 0.945 | 5.5% |

These are tokenizer comparisons, not experiments with three generations of
LLM estimators or reconstructions of historical API bills. The post-hoc
[split-sensitivity analysis](pipeline/results/post_hoc/split_sensitivity_summary.json)
also remains exploratory; it does not replace the original failed decision.

Overall, the study supports a token-count gap and tokenizer sensitivity on
this corpus. It supports neither the simple “equal bits, encoding only”
pattern nor the opposite conclusion that Korean has higher intrinsic entropy.
No second-corpus or alternative-estimator replication is reported here.

## 7. Recomputing the archived statistics

The repository includes text-free per-pair measurements, so the published
statistics can be recomputed without downloading corpus text or model
weights. From a clone of the repository:

```sh
cd pipeline
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m klen.analyze \
  spec/measurement_spec.yaml \
  results/confirmatory/per_pair_full_no_text.csv \
  out/reproduction
```

This uses the original frozen rules and writes to a separate directory.
It reanalyzes **archived measurements**; it does not independently repeat
Qwen scoring or provide new confirmatory evidence. The original environment
used Python 3.13.1; package versions are recorded in the lockfile and provenance.

Optional implementation checks:

```sh
.venv/bin/python -m pytest
```

The suite contains **66 automated checks**, including unit and integration
tests. Three tests may download a tiny GPT-2 checkpoint; one MPS-specific
test is skipped if MPS is unavailable. Test count is not a pass count, and
software checks do not validate the scientific hypothesis.

### Repeating model scoring on a new machine

Full scoring requires additional preparation; it is not turnkey from the
frozen spec, which preserves absolute paths from the original machine:

1. Obtain the [pinned OPUS archive](https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-ko.txt.zip)
   as `pipeline/data/raw/OPUS-GlobalVoices-v2018q4-en-ko.txt.zip`. Then run
   `scripts/prepare_globalvoices.py` from `pipeline/`. The script verifies
   the archive hash and prepares the split; it does **not** download the
   archive or overwrite existing prepared files.
2. Prepare the pinned estimator/tokenizer assets. Model weights and the local
   `MODEL_MANIFEST.json` are not included in Git. For a separate reproduction
   spec, `estimator.local_path: null` allows loading the pinned Hub revision
   instead of the original local snapshot.
3. Make a **separate reproduction specification**, documenting path or loader
   changes and its new hash. Preserve the original spec, freeze timestamp,
   and archived results. A new freeze cannot recreate the original blind
   evaluation.
4. Score to a new output directory with `klen.run_pipeline`, then analyze
   those measurements with `klen.analyze`. Both MPS and CPU are supported;
   CPU may be slower, and numerical results need not be byte-identical across
   hardware. Only confirmatory analysis verifies the frozen content hash;
   the scoring command checks the frozen flag.

The [technical README](pipeline/README_TECHNICAL.md) contains implementation
details but also legacy status notes; its references to unfilled thresholds
and 63 tests describe an earlier project state. The frozen specification
and archived provenance are the record of the completed run.

## Repository layout

```
pipeline/klen/          measurement package: data validation, NLL scoring, token counting,
                        pooled-ratio bootstrap, spec freezing, provenance, analysis CLI
paper/                  paper PDF, LaTeX source, and bibliography
pipeline/tests/         automated checks incl. exact-bit fake models and bootstrap identities
pipeline/spec/          frozen held-out spec, pilot spec, FLORES+ spec, blank template
pipeline/scripts/       corpus preparation/split, result export, README result rendering
pipeline/results/       analysis outputs (summaries, bootstrap replicates, provenance,
                        per-pair measurements with text columns removed)
flores_plus_corpus/     downloader/aligner for the planned gated FLORES+ replication
```

Corpus text and model weights are not redistributed here. The OPUS archive
retains the original sources' licenses; see the [data guide](pipeline/data/README.md)
for licensing and attribution details. Qwen3 and Polyglot-Ko assets use
Apache-2.0. FLORES+ access must be obtained personally under its access
conditions; see its [preparation guide](flores_plus_corpus/README.md).

## References

* C. E. Shannon, "A Mathematical Theory of Communication," *Bell System
  Technical Journal*, 1948.
* C. E. Shannon, "Prediction and Entropy of Printed English," *Bell System
  Technical Journal*, 1951.
* S. J. Mielke et al., [“What Kind of Language Is Hard to Language-Model?”](https://aclanthology.org/P19-1491/),
  *ACL*, 2019.

Further references and limitations are in the [paper](paper/main.pdf) and
its [bibliography](paper/refs.bib).
