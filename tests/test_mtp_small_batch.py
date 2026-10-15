import pytest


def test_completion_gate_rejects_short_equal_prefix_and_first_difference():
    from bench_mtp_small_batch import completion_check

    assert completion_check([3, 4], [3, 4], 2) == {
        "strict_pass": True,
        "first_difference": None,
    }
    assert not completion_check([3, 4], [3], 2)["strict_pass"]
    failure = completion_check([3, 4], [3, 5], 2)
    assert failure == {"strict_pass": False, "first_difference": 1}


@pytest.mark.parametrize("length", [64, 256, 1024])
def test_synthetic_cases_keep_requested_input_and_output_lengths(length):
    from bench_mtp_small_batch import make_cases

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [2, 3, 4]

        def apply_chat_template(self, messages, **kwargs):
            return [7, 8, 9, 10]

    cases = make_cases(Tokenizer())
    selected = [c for c in cases if c["name"].startswith(f"synthetic-{length}-")]
    assert sorted(c["output"] for c in selected) == [128, 512]
    assert all(len(c["prompt"]) == length for c in selected)
    assert {c["name"] for c in cases if not c["name"].startswith("synthetic")} == {
        "chinese",
        "code",
    }


def test_chat_cases_are_token_lists_with_current_transformers_defaults():
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from bench_mtp_small_batch import make_cases

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
        chat_template="{{ messages[0]['content'] }}",
    )
    cases = make_cases(tokenizer)
    assert all(isinstance(c["prompt"], list) for c in cases)
    json.dumps(cases)
