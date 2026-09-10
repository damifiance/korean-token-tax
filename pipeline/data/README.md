# Corpus data

## Selected source

The immediately available corpus is the official OPUS packaging of Global
Voices `v2018q4`, English–Korean, in Moses format.

- Archive: `raw/OPUS-GlobalVoices-v2018q4-en-ko.txt.zip`
- Retrieval and license record:
  `raw/OPUS-GlobalVoices-v2018q4-en-ko.source.json`
- Preparation command: `python3 scripts/prepare_globalvoices.py`
- Pilot file: `processed/globalvoices_v2018q4_en_ko_pilot.csv`
- Held-out confirmatory file:
  `processed/globalvoices_v2018q4_en_ko_confirmatory.csv`
- Generated validation manifest:
  `processed/globalvoices_v2018q4_en_ko_split_manifest.json`

The source archive contains 365 article-level XML alignment groups. Eighteen
groups contain no aligned text, leaving 347 article clusters in the analysis
file. Its two plain-text files contain a blank separator after every group, so
the OPUS API reports 9,382 lines while the usable aligned-text count is 9,017.
The preparation script checks this relationship and assigns every alignment
from the same nonempty XML `linkGrp` to one `cluster_id`.

`translation_direction` is `unknown`: the archive establishes alignment but
does not provide per-article translation-direction metadata. Because the
release has no train/dev/test division, the script creates a deterministic
article-level split. First, article clusters connected by any exact bilingual
duplicate are placed in one indivisible component. Components are ranked by a
seeded SHA-256 rule, and the prefix whose article count is closest to 20% is
assigned to the pilot. The seed is `20260910`; the result is 69 pilot articles
(1,732 rows) and 278 held-out articles (7,285 rows). No article or exact
bilingual pair crosses the split. Do not inspect or measure the confirmatory
file until the thresholds and endpoints are frozen.

The script applies only Unicode NFC normalization and removal of surrounding
whitespace. It retains the original OPUS document paths and `xtargets` alignment
metadata in audit columns.

## License and provenance caveat

The archive's own `LICENSE` file says the data retains the licenses of the
original sources, without naming an SPDX identifier. The current Global Voices
site displays CC BY 3.0, while OPUS disclaims ownership. Keep the citation,
article URL, and attribution information with any reported examples, and check
individual historical pages before republishing their full text.

The corpus was released in 2018. A modern language model may have encountered
Global Voices or this OPUS package during training. Unless a model provider
documents its training data sufficiently to rule that out, contamination must
be recorded as unknown rather than absent.
