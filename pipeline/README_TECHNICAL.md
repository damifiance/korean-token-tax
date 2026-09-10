# klen pipeline — technical usage notes

Measurement pipeline for model-relative code length (bits) vs. billing tokens
on an aligned Korean-English parallel corpus. This document covers operation
only; all research decisions live in `spec/measurement_spec.yaml`.

## Environment

- Python venv at `.venv/` (Python 3.13.1), exact versions in `requirements.lock.txt`.
- Recreate: `python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt`
- Apple Silicon: pass `--device mps` for scoring; log-softmax is always float32.

### Project-local estimator snapshot

The pinned Qwen checkpoint is already present at
`models/Qwen3-1.7B-Base-ea980cb/`. Keep the canonical Hub identity in the
measurement spec and use `local_path` only as an offline loading source:

```yaml
estimator:
  model_id: Qwen/Qwen3-1.7B-Base
  revision: ea980cb0a6c2ae4b936e82123acc929f1cec04c1
  local_path: ../models/Qwen3-1.7B-Base-ea980cb
```

Relative paths are resolved from the directory containing the spec. Before
loading, `MODEL_MANIFEST.json` is checked against the canonical model ID and
revision, every downloaded file's size and SHA-256 are verified, and the local
model/tokenizer configuration is checked against the recorded Qwen identity.
The canonical ID and revision, local load source, and tokenizer hashes remain
in each provenance JSON. This prevents an offline path from silently replacing
the frozen estimator with a different or incomplete checkpoint.

## Workflow

1. Run the reserved pilot only through `spec/pilot_spec.yaml`. To score all
   1,732 pilot rows while retaining the pilot output tag:
   `.venv/bin/python -m klen.run_pipeline spec/pilot_spec.yaml out/pilot --allow-unfrozen --pilot 1732 --device mps`
2. Compute exploratory pilot estimates and variance:
   `.venv/bin/python -m klen.analyze spec/pilot_spec.yaml out/pilot/per_pair_pilot.csv out/pilot --exploratory`
3. Use the pilot to make your own threshold, endpoint and exclusion decisions,
   then fill the seven remaining `null`/empty fields in
   `spec/measurement_spec.yaml` (and, if wanted, the `exclusions` block).
   Do not inspect or measure the held-out corpus while doing this.
   `out/pilot/precision_sim_summary.json` reports, from the pilot cluster
   sums, the distribution of the bootstrap bounds the confirmatory rules
   will use at 278 held-out clusters (no thresholds are proposed there).
4. Freeze the held-out specification (records a content hash and refuses if
   anything remains missing):
   `.venv/bin/python -m klen.freeze_spec spec/measurement_spec.yaml`
5. Full held-out run:
   `.venv/bin/python -m klen.run_pipeline spec/measurement_spec.yaml out/confirmatory --device mps`
6. Confirmatory analysis (requires frozen, untampered spec; refuses pilot files):
   `.venv/bin/python -m klen.analyze spec/measurement_spec.yaml out/confirmatory/per_pair_full.csv out/confirmatory`

`spec/measurement_spec_template.yaml` is the untouched blank template for a
future study. The current Global Voices configuration is already populated in
the other two spec files; thresholds and endpoints remain deliberately unset.

## Endpoints and the overall decision

`endpoints.confirmatory` names the comparisons that count (any of `R_T`,
`R_B_equivalence`, `R_eta`, `C_K`); `endpoints.decision_rule` combines them
(`all_confirmatory_pass` is the only implemented rule). Confirmatory analysis
writes `confirmatory_decisions` (listed comparisons only), `exploratory_decisions`
(the other comparisons, evaluated against the same frozen thresholds but
outside the rule) and `overall_decision`. Endpoint names and the rule are
validated before freezing. Freezing rewrites the spec canonically, which drops
YAML comments; `spec/measurement_spec_template.yaml` keeps the annotated form.

## Exclusion rules

`exclusions` in the spec (all `null` by default = analyse every scored row)
declares which flagged rows are dropped at analysis time: identical
Korean/English text (`drop_ko_equals_en`, untranslated captions), no Hangul on
the Korean side, disallowed control characters, minimum character counts, and
a character-ratio window. Measurement always scores every row; `analyze`
applies the rules, records counts dropped per rule in `analysis_summary.json`,
and the block is part of the frozen spec content, so rules cannot change
after freezing. Unknown rule names are an error. Pilot sensitivity (2026-09-10):
every combination of these rules moved the pilot pooled ratios by less than
0.03.

## Exploratory section of `analyze`

Every `analysis_summary.json` now also carries an `exploratory` object that
never feeds the confirmatory rules: per-pair ratio distribution (mean, median,
geometric mean, quantiles — descriptive only, the estimand stays the pooled
ratio-of-sums), pooled estimands with cluster-bootstrap CIs per genre and per
length stratum (quantile bins of `exploratory.length_variable`, default
`n_chars_en`), and paired cluster-level sign-flip permutation tests for
R_T, R_B, R_eta (H0: language labels exchangeable within pairs) and C_K (H0:
tokenizer labels exchangeable). `--skip-exploratory-section` omits it.

## FLORES+ (second corpus, not yet downloaded)

`spec/flores_plus_spec.yaml` mirrors the Global Voices decisions and points at
`../flores_plus_corpus/processed/flores_plus_ko_en.csv`. The corpus is gated;
see `../flores_plus_corpus/README.md` for the token procedure (the account
holder must accept the conditions personally). Its role — exploratory
multi-genre replication versus second confirmatory corpus — is a research
decision recorded in that file's header comment. Run it with the same
commands, substituting the spec path; it has no pilot/held-out split.

## Tests

`.venv/bin/python -m pytest` — tests cover NFC/validation, exact-bit scoring
against hand-computable fake models, chunked non-overlapping target accounting,
BOS/EOS policies, pooled-ratio identities (R_eta = R_B/R_T in every bootstrap
replicate), whole-cluster resampling, stratum preservation, equivalence-test
mechanics, spec freeze/tamper detection, exclusion rules, stratified
estimates, and the sign-flip permutation test (63 tests). `-m "not model"`
skips the three tests that download a tiny random GPT-2.

## Counting-tokenizer identifiers

- Hugging Face: use the repository ID and immutable commit SHA, for example
  `EleutherAI/polyglot-ko-1.3b` at
  `557e162cf6e944fdbae05bab2e45d066a125eacb`.
- OpenAI tiktoken: use `tiktoken:<encoding>` as the ID and the exact installed
  package version as the revision, for example `tiktoken:o200k_base` at
  `0.14.0`. The official vocabulary asset is cached under
  `artifacts/tokenizers/tiktoken-cache/` and SHA-256 verified before use.
- Raw tiktoken encodings do not add default wrapper tokens. They support
  `content_only` and `with_special_tokens` (equal counts), but not
  `chat_template` accounting.
- See `artifacts/tokenizers/TOKENIZER_MANIFEST.json` for immutable source
  revisions, file hashes, vocabulary sizes, normalizers, and special-token
  behavior.

## Guarantees enforced in code

- Estimator model is only ever paired with its own tokenizer;
  counting tokenizers return integers only, never token IDs for scoring.
- `add_special_tokens=False` everywhere in scoring; BOS/EOS added explicitly
  per the declared policy; scored-target count asserted every call.
- For Qwen3 Base, `tokenizer_config.json` intentionally disables automatic BOS
  insertion, while the pinned model and generation configurations declare ID
  151643 (`<|endoftext|>`) as `bos_token_id`. Qwen's training documentation
  ([control-token concepts](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md))
  identifies that token as the separator between packed documents. Therefore
  `prepend_bos` means “condition this isolated message on a document boundary”
  and gives a code length for every content token. The loader synchronizes the
  tokenizer's missing BOS field from the model configuration only; it never
  overrides an existing tokenizer BOS. Use `eos: none` if no artificial
  post-message boundary is to be charged.
- Confirmatory analysis is blocked until the spec is frozen, and blocked again
  if the spec content changes after freezing (SHA-256 check).
- Bootstrap resamples complete clusters within genre strata, pairs never split.
- Provenance JSON records package versions, tokenizer file hashes, dataset
  hash, spec, device, and validation report for every run.

## Interpretation guards (carried in every analysis output)

B_M is model-relative surprisal, not semantic information or intrinsic
entropy. R_T, R_B, R_eta are functionally dependent. 1 − R_eta is a relative
shortfall under an English-efficiency counterfactual. C_K < 1 measures an
absolute Korean-token reduction under the comparison tokenizer. G is retained
as exploratory gap contraction only: it can fall because English token count
rises even if Korean token count does not fall. Causal attribution requires
controlling tokenizer algorithm, vocab size, normalization, pre-tokenization,
and training-corpus size.
