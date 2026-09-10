# FLORES+ Korean–English corpus preparation

This directory contains a reproducible downloader and aligner for the Korean and
English `dev` and `devtest` portions of FLORES+. It pins the upstream repository
to a commit instead of following the changing `main` branch.

## Frozen source

- Repository: `openlanguagedata/flores_plus`
- Dataset version shown by the upstream card: `4.6`
- Revision: `5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06`
- Language configurations: `eng_Latn`, `kor_Hang`
- Splits: `dev`, `devtest`
- License: CC BY-SA 4.0
- Expected rows: 997 `dev` pairs and 1,012 `devtest` pairs (2,009 total)
- Source page: <https://huggingface.co/datasets/openlanguagedata/flores_plus>

## Access condition

The files are auto-gated by Hugging Face. The account holder must personally
open the source page, accept its current conditions, and obtain a read token.
Do not accept the conditions unless you can truthfully satisfy them. In
particular, the upstream gate asks evaluators to ensure that the benchmark was
not in the evaluated model's training data. That may not be demonstrable for an
off-the-shelf language model with undisclosed training data.

The dataset card also asks users not to expose local copies to web crawlers.
Consequently, `raw/`, `processed/`, and `manifest.json` are ignored here. Do not
publish them in a public repository.

## Run after access has been approved

Enter the token without putting it in a command-line argument or source file:

```zsh
read -s "HF_TOKEN?Hugging Face read token: "
export HF_TOKEN
python3 prepare_flores_plus.py
unset HF_TOKEN
```

The script downloads only four source files at the frozen revision, verifies
the documented row counts and one-to-one alignment, and creates:

- `processed/flores_plus_ko_en.jsonl`
- `processed/flores_plus_ko_en.csv`
- `manifest.json`

The two processed files have these requested columns:

```text
pair_id, cluster_id, genre, translation_direction,
ko_text, en_text, source_url, split
```

They also retain `flores_id`, `domain`, `topic`, and both language files'
`last_updated` values. `cluster_id` is a deterministic hash of the source URL,
so sentences drawn from the same source document remain in the same bootstrap
cluster. `genre` is the upstream `domain` (`wikinews`, `wikijunior`, or
`wikivoyage`), and `translation_direction` is recorded as `en->ko` based on the
dataset card's description of the collection.

`manifest.json` records SHA-256 hashes, byte sizes, row counts, genre counts,
cluster count, and the exact upstream revision. Preserve it with the private
analysis data.

