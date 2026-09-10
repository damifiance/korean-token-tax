# Why does Korean pay more tokens? (한국어의 "인코딩 세금" 검증)

Measuring whether Korean's higher LLM token cost comes from *more information*
or from *worse encoding*, using Shannon's code-length view of tokenization on
an aligned Korean–English corpus, with the decision rules fixed before the
final data was opened.

---

## 1. The puzzle

Send the same message to a language model in Korean and in English. The
Korean version is billed for noticeably more tokens. Everyone who has used an
LLM API from Korea has noticed this; the usual reaction is "the tokenizer is
bad at Korean". But that is only one of two possible explanations, and they
predict different things:

| | **A. Information hypothesis** | **B. Encoding hypothesis** |
|---|---|---|
| Why more tokens? | The Korean text really carries more information (it says more, or is harder to predict). | The tokenizer, trained mostly on English, chops Korean into small pieces. |
| Token ratio KR/EN | high | high |
| Bit ratio KR/EN | high, similar to the token ratio | close to 1 |
| Korean-trained tokenizer | small change | Korean token count drops |

The two hypotheses agree about token counts, so counting tokens cannot
separate them. They disagree about *information*, so we need a way to measure
information that does not depend on the tokenizer under test.

## 2. The intuition: tokenizers are codes, and a model is a codebook

Shannon's insight is that "how much information a message contains" and "how
many symbols it takes to write down" are two different things, connected by a
*code*. A good code spends few symbols on predictable content and more on
surprising content. A bad code wastes symbols.

A tokenizer is exactly such a code: it turns text into a sequence of symbols
(tokens). And a language model is, in effect, a codebook that tells us how
surprising each piece of text is. Arithmetic coding makes this concrete: a
model that assigns probability `P` to the next token can encode that token in
`-log2 P` bits. Summed over a message, this gives its **model-relative code
length**

```
B_M(s) = - Σ_j log2 P_M(z_j | z_<j),      z = τ_M(s)
```

which is measured in bits and does *not* care how the billing tokenizer
splits the text. It answers "how much does this model have to be told to
reproduce this message?"

With bits in hand, the billing tokenizer's efficiency is simply

```
η(s) = B_M(s) / T_b(s)        bits carried per billed token
```

Now the two hypotheses separate cleanly. Corpus-wide, define

```
R_T = Σ T_KR / Σ T_EN        token ratio
R_B = Σ B_KR / Σ B_EN        bit ratio
R_η = R_B / R_T              efficiency ratio  (bits/token in Korean ÷ bits/token in English)
```

* If Korean genuinely carries more information, bits rise with tokens:
  `R_B ≈ R_T` and `R_η ≈ 1`.
* If Korean is merely encoded badly, tokens rise but bits do not:
  `R_T ≫ 1`, `R_B ≈ 1`, and `R_η < 1`. The shortfall `1 − R_η` is the
  "encoding tax": how much less information each Korean token carries than
  an English token would, under this model. It is a relative shortfall, not
  a literal count of empty tokens.

A second, independent check: re-encode the *same* Korean sentences with a
tokenizer trained on Korean. The text and its information are unchanged; only
the code changes. If the Korean token count falls (`C_K < 1`), the original
cost was at least partly a property of the code, not the message.

### What the bits do and do not mean

`B_M` is *model-relative* surprisal, not intrinsic entropy. Writing the
model's distribution as `Q` and the true one as `P`,
`H(P,Q) = H(P) + D_KL(P‖Q)`: the measured bits include the model's own
prediction error. If the estimator model knows Korean less well than English,
Korean bits are inflated, which biases *against* the encoding hypothesis
(it makes `R_B` larger). Parallel meaning also does not guarantee equal
information: a translation may add or drop content. The pipeline records
these guards in every output file so they travel with the numbers.

## 3. Pre-registration: deciding what "well above one" means before looking

Ratios like "≫ 1" and "≈ 1" are meaningless until thresholds are fixed, and
thresholds chosen after seeing the data prove nothing. The study therefore
splits the corpus into a **pilot** (used for variance estimates and for
choosing thresholds) and a **held-out** set that was scored only after the
specification was frozen. Freezing writes a SHA-256 of the spec content into
the file; the analysis refuses to run on an unfrozen or edited spec, and
refuses pilot-tagged inputs in confirmatory mode.

Frozen decisions (`pipeline/spec/measurement_spec.yaml`, hash `850fa9be…`):

| Decision | Value | Rule applied to the held-out bootstrap |
|---|---|---|
| Primary endpoint | `R_η` | one-sided 95 % upper bound < `c_η = 0.95` |
| Token ratio | `R_T` | one-sided 95 % lower bound > `c_T = 1.20` |
| Tokenizer control | `C_K` | one-sided 95 % upper bound < `c_C = 0.90` |
| Overall | all three must pass | `all_confirmatory_pass` |
| Bit-ratio equivalence (exploratory) | `R_B` | 90 % CI inside `[0.8333, 1.20]` |
| Exclusions | drop untranslated rows (Korean text identical to English) and rows with no Hangul on the Korean side | applied identically to pilot and held-out |

Everything else (per-genre and per-length breakdowns, permutation tests,
per-pair distributions, the gap-contraction statistic `G`, and the FLORES+
replication) is exploratory and is labelled as such in the output.

## 4. Setup

| Component | Choice | Pinned identity |
|---|---|---|
| Corpus | OPUS Global Voices v2018q4 en–ko, article-aligned news | 347 articles, 9,017 pairs; deterministic article-level split (seed 20260910): pilot 69 articles / 1,732 pairs, held-out 278 / 7,285 |
| Estimator model `M` | Qwen3-1.7B-Base | revision `ea980cb0…`, bf16 weights, float32 log-softmax, MPS |
| Baseline billing tokenizer `b` | OpenAI `o200k_base` (tiktoken 0.14.0) | vocabulary asset SHA-256 verified |
| Korean-trained comparison tokenizer | EleutherAI polyglot-ko-1.3b | revision `557e162c…`, 30k vocabulary, Korean-only training corpus |
| Token accounting | content only (no chat wrapper) | |
| Scoring policy | prepend the model's document boundary token, score every content token, no EOS | |
| Statistics | ratio-of-sums estimands; paired cluster bootstrap (whole articles resampled, both languages together), 10,000 replicates | seed 20260910 |

Every pooled ratio is a ratio of corpus sums, never a mean of per-message
ratios; the identity `R_η = R_B / R_T` therefore holds exactly in every
bootstrap replicate, and the pipeline asserts it.

## 5. Results

<!-- RESULTS:BEGIN -->
Held-out set after exclusions: **7,046 pairs in 278 articles** (239 rows dropped: drop_ko_equals_en 104, drop_ko_no_hangul 135). Pilot after the same rules: 1,672 pairs in 69 articles.

### Pooled estimands (held-out, 95 % cluster-bootstrap percentile CI)

| Quantity | Point | 95 % CI |
|---|---|---|
| `R_T` token ratio KR/EN (o200k) | 1.3888 | [1.3714, 1.4058] |
| `R_B` bit ratio KR/EN (Qwen3-1.7B) | 1.3121 | [1.2959, 1.3278] |
| `R_η` = `R_B` / `R_T` | 0.9448 | [0.9363, 0.9532] |
| `1 − R_η` encoding shortfall | 0.0552 | [0.0468, 0.0637] |
| `C_K` Korean tokens, polyglot-ko / o200k | 0.8665 | [0.8601, 0.8729] |
| `R_T` under polyglot-ko (exploratory) | 0.5095 | [0.5025, 0.5163] |
| `G` gap contraction (exploratory) | 0.3668 | [0.3629, 0.3707] |

### Pre-registered decisions

| Comparison | Rule | Bound observed | Result |
|---|---|---|---|
| `R_eta` | one-sided 95% upper bound < c_eta=0.95 | 0.9519 | FAIL |
| `R_T` | one-sided 95% lower bound > c_T=1.2 | 1.3741 | PASS |
| `C_K` | one-sided 95% upper bound < c_C=0.9 | 0.8719 | PASS |

**Overall (all_confirmatory_pass): FAIL.**

The pre-registered rule was not met because `R_eta` (bound 0.9519 vs threshold in rule "one-sided 95% upper bound < c_eta=0.95"; point estimate 0.9448). The thresholds were frozen before this data was scored and are not revised here; the point estimates and intervals above are reported as observed.

Exploratory bit-ratio equivalence: 90 % CI for `R_B` = [1.2984, 1.3252] against [0.8333, 1.20] → not inside the interval. Korean text costs more bits under this model as well as more tokens; the pre-registered primary question is whether bits rise *as fast as* tokens, which is what `R_η` measures.

### Pilot vs held-out

| Quantity | Pilot (69 articles) | Held-out (278 articles) |
|---|---|---|
| `R_T` | 1.3769 | 1.3888 |
| `R_B` | 1.2943 | 1.3121 |
| `R_η` | 0.9400 | 0.9448 |
| `C_K` | 0.8697 | 0.8665 |

### By message length (exploratory; terciles of `n_chars_en`)

| Stratum | Range (chars) | n | `R_T` | `R_B` | `R_η` [95 % CI] | `C_K` |
|---|---|---|---|---|---|---|
| Q1 | 3–74 | 2349 | 1.564 | 1.419 | 0.907 [0.894, 0.921] | 0.882 |
| Q2 | 74–137 | 2348 | 1.415 | 1.318 | 0.931 [0.919, 0.944] | 0.868 |
| Q3 | 137–587 | 2349 | 1.329 | 1.271 | 0.956 [0.946, 0.966] | 0.861 |

### Paired cluster permutation tests (exploratory; 10,000 sign-flips of whole articles)

| Quantity | log pooled ratio | two-sided p |
|---|---|---|
| `R_T` | +0.3284 | 0.00010 |
| `R_B` | +0.2717 | 0.00010 |
| `R_eta` | -0.0567 | 0.00010 |
| `C_K` | -0.1433 | 0.00010 |

Minimum attainable p is 1/(1+10,000); the three language ratios are one dependent finding.

### Per-message ratio distribution (descriptive only; not the estimand)

| Ratio | mean | median | geometric mean | 5th pct | 95th pct |
|---|---|---|---|---|---|
| `R_T` | 1.489 | 1.429 | 1.402 | 0.762 | 2.364 |
| `R_B` | 1.396 | 1.357 | 1.331 | 0.766 | 2.091 |
| `R_eta` | 0.973 | 0.944 | 0.949 | 0.666 | 1.369 |

Full outputs, including 10,000 bootstrap replicates, provenance (package versions, tokenizer file hashes, dataset hash, device) and the pilot precision simulation, are in `pipeline/results/`.
<!-- RESULTS:END -->

## 6. Reading the numbers honestly

* `R_T`, `R_B`, `R_η` are functionally dependent; they are one finding seen
  from three angles, not three findings.
* `1 − R_η` is a shortfall relative to an English-efficiency counterfactual,
  not a literal count of "empty" tokens.
* `C_K < 1` shows that a different code encodes the same Korean in fewer
  symbols. Attributing that to the training-language mix alone would require
  holding the algorithm, vocabulary size, normalisation and pre-tokenisation
  fixed, which two off-the-shelf tokenizers do not.
* Global Voices predates the estimator model; training-data contamination
  is unknown, not excluded.
* Bits are Qwen3-1.7B-relative. A different estimator gives different bits;
  the exploratory replications are there to see whether the *ratios* move.

## 7. Reproducing

```zsh
cd pipeline
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
.venv/bin/python -m pytest                      # 66 tests, hand-verifiable NLL cases included

python3 scripts/prepare_globalvoices.py         # builds the pilot / held-out split from the OPUS archive
.venv/bin/python -m klen.run_pipeline spec/pilot_spec.yaml out/pilot --allow-unfrozen --pilot 1732 --device mps
.venv/bin/python -m klen.analyze spec/pilot_spec.yaml out/pilot/per_pair_pilot.csv out/pilot --exploratory

.venv/bin/python -m klen.freeze_spec spec/measurement_spec.yaml
.venv/bin/python -m klen.run_pipeline spec/measurement_spec.yaml out/confirmatory --device mps
.venv/bin/python -m klen.analyze spec/measurement_spec.yaml out/confirmatory/per_pair_full.csv out/confirmatory
```

The estimator snapshot and tokenizer files are fetched at the pinned
revisions; `MODEL_MANIFEST.json` and `TOKENIZER_MANIFEST.json` carry file
hashes that are re-verified before every run. Operational details are in
`pipeline/README_TECHNICAL.md`.

## Repository layout

```
pipeline/klen/          measurement package: data validation, NLL scoring, token counting,
                        pooled-ratio bootstrap, spec freezing, provenance, analysis CLI
pipeline/tests/         unit tests incl. exact-bit fake models and bootstrap identity checks
pipeline/spec/          frozen held-out spec, pilot spec, FLORES+ spec, blank template
pipeline/scripts/       Global Voices download/alignment/split
pipeline/results/       analysis outputs (summaries, bootstrap replicates, provenance,
                        per-pair measurements with text columns removed)
flores_plus_corpus/     downloader/aligner for the gated FLORES+ multi-genre replication
```

Corpus text is not redistributed here. Global Voices is CC BY 3.0 via OPUS
and can be rebuilt with the script; FLORES+ is gated and must be obtained
personally. The Qwen3 weights (Apache-2.0) and polyglot-ko tokenizer
(Apache-2.0) are downloaded at the pinned revisions.

## References

* C. E. Shannon, "A Mathematical Theory of Communication," *Bell System
  Technical Journal*, 1948.
* C. E. Shannon, "Prediction and Entropy of Printed English," *Bell System
  Technical Journal*, 1951.
