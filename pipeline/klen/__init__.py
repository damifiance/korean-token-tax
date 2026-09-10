"""klen — Korean/English model-relative code-length measurement pipeline.

Measures, for aligned Korean-English message pairs:
  B_M(s)  summed autoregressive NLL in bits under a fixed estimator model M
  T_b(s)  token counts under billing tokenizers b
and the pooled corpus ratios R_T, R_B, R_eta = R_B / R_T.

B_M is model-relative surprisal (arithmetic-code length under M), not semantic
information and not intrinsic language entropy.
"""

__version__ = "0.1.0"
