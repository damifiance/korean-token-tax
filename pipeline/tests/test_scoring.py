import json
import math
from types import SimpleNamespace

import pytest
import torch

from klen import scoring
from klen.scoring import ScoringPolicy

VOCAB = 10


def make_local_estimator_snapshot(tmp_path, model_id="example/model",
                                  revision="exact-revision"):
    root = tmp_path / "snapshot"
    root.mkdir()
    payloads = {
        "config.json": {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 8,
            "max_position_embeddings": 32,
            "bos_token_id": 7,
            "eos_token_id": 7,
            "torch_dtype": "bfloat16",
        },
        "tokenizer.json": {
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
            "normalizer": {"type": "NFC"},
            "added_tokens": [{"id": 7, "content": "<|endoftext|>"}],
        },
        "tokenizer_config.json": {"add_bos_token": False},
    }
    for name, obj in payloads.items():
        (root / name).write_text(json.dumps(obj), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"test weights")

    files = {}
    for path in sorted(root.iterdir()):
        files[path.name] = {
            "size": path.stat().st_size,
            "sha256": scoring._file_sha256(path),
        }
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "files": files,
        "identity": {
            "model": payloads["config.json"],
            "tokenizer": {
                "model_type": "BPE",
                "normalizer_type": "NFC",
                "regular_vocab_size": 2,
                "endoftext_token": "<|endoftext|>",
                "endoftext_token_id": 7,
                "add_bos_token": False,
            },
        },
    }
    (root / "MODEL_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class FakeTokenizer:
    """Maps each character to a token id in [2, VOCAB); bos=0, eos=1."""
    bos_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [2 + (ord(c) % (VOCAB - 2)) for c in text]}


class _Out:
    def __init__(self, logits):
        self.logits = logits


class UniformModel:
    """Uniform next-token distribution: every target costs exactly log2(VOCAB) bits."""
    def eval(self):
        return self

    def __call__(self, input_ids):
        L = input_ids.shape[1]
        return _Out(torch.zeros(1, L, VOCAB, device=input_ids.device))


class BiasedModel:
    """Puts logit ln(9) on token id 3, 0 elsewhere: p(3) = 9/18 = 0.5 exactly,
    so a target token 3 costs exactly 1 bit."""
    def eval(self):
        return self

    def __call__(self, input_ids):
        L = input_ids.shape[1]
        logits = torch.zeros(1, L, VOCAB)
        logits[:, :, 3] = math.log(9.0)
        return _Out(logits)


class ThreeTokenizer(FakeTokenizer):
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3] * len(text)}


def test_uniform_model_exact_bits_bos_policy():
    tok = FakeTokenizer()
    policy = ScoringPolicy(bos="prepend_bos", eos="none")
    res = scoring.score_text(UniformModel(), tok, "abcde", policy)
    # 5 content tokens, all scored (first from BOS): 5 * log2(10)
    assert res.n_content_tokens == 5
    assert res.n_targets == 5
    assert res.bits_total == pytest.approx(5 * math.log2(VOCAB), rel=1e-6)


def test_uniform_model_exact_bits_no_bos_skip_first():
    tok = FakeTokenizer()
    policy = ScoringPolicy(bos="none_skip_first", eos="none")
    res = scoring.score_text(UniformModel(), tok, "abcde", policy)
    # first content token unscored: 4 * log2(10)
    assert res.n_targets == 4
    assert res.bits_total == pytest.approx(4 * math.log2(VOCAB), rel=1e-6)


def test_eos_policy_adds_one_scored_target():
    tok = FakeTokenizer()
    res_no = scoring.score_text(
        UniformModel(), tok, "abc", ScoringPolicy(bos="prepend_bos", eos="none"))
    res_eos = scoring.score_text(
        UniformModel(), tok, "abc", ScoringPolicy(bos="prepend_bos", eos="append_scored"))
    assert res_eos.n_targets == res_no.n_targets + 1
    assert res_eos.bits_total == pytest.approx(
        res_no.bits_total + math.log2(VOCAB), rel=1e-6)
    # content-token count is unaffected by EOS
    assert res_eos.n_content_tokens == res_no.n_content_tokens == 3


def test_biased_model_exactly_one_bit_per_target():
    tok = ThreeTokenizer()
    policy = ScoringPolicy(bos="prepend_bos", eos="none")
    res = scoring.score_text(BiasedModel(), tok, "xxxx", policy)  # 4 tokens of id 3
    assert res.n_targets == 4
    assert res.bits_total == pytest.approx(4.0, rel=1e-6)


def test_chunked_scoring_counts_every_target_exactly_once():
    tok = FakeTokenizer()
    text = "abcdefghijklmnop"  # 16 content tokens (+ BOS = 17)
    for W, overlap in [(4, 1), (4, 2), (5, 3), (8, 4)]:
        policy = ScoringPolicy(bos="prepend_bos", eos="none",
                               context_length=W, context_overlap=overlap)
        res = scoring.score_text(UniformModel(), tok, text, policy)
        assert res.n_targets == 16
        assert res.n_chunks > 1
        # Uniform model is context-independent, so bits depend only on the
        # number of scored targets: any double/missed scoring changes the sum.
        assert res.bits_total == pytest.approx(16 * math.log2(VOCAB), rel=1e-6)


def test_chunking_noop_when_sequence_fits():
    tok = FakeTokenizer()
    p_plain = ScoringPolicy(bos="prepend_bos", eos="none")
    p_chunk = ScoringPolicy(bos="prepend_bos", eos="none",
                            context_length=100, context_overlap=10)
    r1 = scoring.score_text(UniformModel(), tok, "abcde", p_plain)
    r2 = scoring.score_text(UniformModel(), tok, "abcde", p_chunk)
    assert r1.bits_total == pytest.approx(r2.bits_total, rel=1e-12)
    assert r2.n_chunks == 1


def test_empty_text_raises():
    with pytest.raises(ValueError, match="zero content tokens"):
        scoring.score_text(UniformModel(), FakeTokenizer(), "",
                           ScoringPolicy(bos="prepend_bos", eos="none"))


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        ScoringPolicy(bos="maybe", eos="none")
    with pytest.raises(ValueError):
        ScoringPolicy(bos="prepend_bos", eos="sometimes")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_moves_bits_to_cpu_before_float64_accumulation():
    """MPS has no float64 support, so `.double()` must happen after `.cpu()`."""
    policy = ScoringPolicy(bos="prepend_bos", eos="none")
    res = scoring.score_text(
        UniformModel(), FakeTokenizer(), "abcde", policy,
        device="mps", keep_per_token=True,
    )
    assert res.n_targets == 5
    assert res.bits_total == pytest.approx(5 * math.log2(VOCAB), rel=1e-6)
    assert len(res.per_token_bits) == 5


@pytest.mark.parametrize(
    ("tokenizer_bos", "config_bos", "expected_bos"),
    [(None, 7, 7), (3, 7, 3), (None, None, None)],
)
def test_load_estimator_synchronizes_only_missing_bos(
    monkeypatch, tokenizer_bos, config_bos, expected_bos,
):
    class LoaderTokenizer:
        def __init__(self):
            self.bos_token_id = tokenizer_bos

    class LoaderModel:
        def __init__(self):
            self.config = SimpleNamespace(bos_token_id=config_bos)
            self.device = None
            self.is_eval = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    tok = LoaderTokenizer()
    model = LoaderModel()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: tok)
    )
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained",
        classmethod(lambda cls, *a, **kw: model),
    )

    loaded_model, loaded_tok = scoring.load_estimator(
        "example/model", "exact-revision", dtype="float32", device="cpu"
    )

    assert loaded_tok.bos_token_id == expected_bos
    assert loaded_tok._klen_source_id == "example/model"
    assert loaded_tok._klen_revision == "exact-revision"
    assert loaded_model.device == "cpu"
    assert loaded_model.is_eval is True


def test_validate_local_estimator_snapshot_checks_identity_and_hashes(tmp_path):
    root = make_local_estimator_snapshot(tmp_path)
    assert scoring.validate_local_estimator_snapshot(
        root, "example/model", "exact-revision"
    ) == root.resolve()

    with pytest.raises(ValueError, match="revision mismatch"):
        scoring.validate_local_estimator_snapshot(
            root, "example/model", "different-revision"
        )

    (root / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        scoring.validate_local_estimator_snapshot(
            root, "example/model", "exact-revision"
        )


def test_load_estimator_uses_verified_local_snapshot(monkeypatch, tmp_path):
    root = make_local_estimator_snapshot(tmp_path)
    calls = []

    class LoaderTokenizer:
        bos_token_id = None

    class LoaderModel:
        config = SimpleNamespace(bos_token_id=7)

        def to(self, device):
            return self

        def eval(self):
            return self

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_tokenizer(cls, source, **kwargs):
        calls.append(("tokenizer", source, kwargs))
        return LoaderTokenizer()

    def load_model(cls, source, **kwargs):
        calls.append(("model", source, kwargs))
        return LoaderModel()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(load_tokenizer))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(load_model))

    _, tok = scoring.load_estimator(
        "example/model", "exact-revision", local_path=root
    )

    assert [kind for kind, _, _ in calls] == ["tokenizer", "model"]
    assert all(source == root.resolve() for _, source, _ in calls)
    assert all(kwargs["local_files_only"] is True for _, _, kwargs in calls)
    assert all("revision" not in kwargs for _, _, kwargs in calls)
    assert tok.bos_token_id == 7
    assert tok._klen_source_id == "example/model"
    assert tok._klen_revision == "exact-revision"
    assert tok._klen_snapshot_dir == str(root.resolve())


# ---------------------------------------------------------------------------
# Real-model consistency tests (tiny random GPT-2, ~few MB; network on first run)
# ---------------------------------------------------------------------------

TINY = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    model.eval()
    return model, tok


def manual_bits(model, ids):
    """Independent reimplementation: single forward, float32 log-softmax,
    position-by-position gather."""
    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for j in range(1, len(ids)):
        total += -logprobs[0, j - 1, ids[j]].item() / math.log(2.0)
    return total


@pytest.mark.model
def test_tiny_gpt2_matches_manual_computation(tiny_model):
    model, tok = tiny_model
    text = "Hello world, 안녕하세요."
    policy = ScoringPolicy(bos="prepend_bos", eos="none")
    res = scoring.score_text(model, tok, text, policy, keep_per_token=True)
    ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
    assert res.full_ids == ids
    assert res.bits_total == pytest.approx(manual_bits(model, ids), rel=1e-5)
    assert len(res.per_token_bits) == res.n_targets == len(ids) - 1
    assert all(b >= 0 for b in res.per_token_bits)


@pytest.mark.model
def test_tiny_gpt2_no_special_tokens_injected(tiny_model):
    model, tok = tiny_model
    text = "test sentence"
    policy = ScoringPolicy(bos="none_skip_first", eos="none")
    res = scoring.score_text(model, tok, text, policy)
    content = tok(text, add_special_tokens=False)["input_ids"]
    assert res.full_ids == content  # nothing prepended/appended
    assert res.n_targets == len(content) - 1


@pytest.mark.model
def test_tiny_gpt2_determinism(tiny_model):
    model, tok = tiny_model
    policy = ScoringPolicy(bos="prepend_bos", eos="append_scored")
    r1 = scoring.score_text(model, tok, "determinism check", policy)
    r2 = scoring.score_text(model, tok, "determinism check", policy)
    assert r1.bits_total == r2.bits_total
