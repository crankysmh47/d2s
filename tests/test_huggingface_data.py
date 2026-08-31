from ppmoe.config import DatasetSpec
from ppmoe.data.huggingface import load_wikitext_test, stream_fineweb


class RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [{"id": "row", "text": "content"}]


def test_fineweb_loader_passes_every_pinned_identity_field() -> None:
    spec = DatasetSpec(
        repo_id="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        revision="fineweb-sha",
        streaming=True,
    )
    loader = RecordingLoader()

    result = stream_fineweb(spec, loader=loader)

    assert list(result) == [{"id": "row", "text": "content"}]
    assert loader.calls == [
        (
            ("HuggingFaceFW/fineweb-edu",),
            {
                "name": "sample-10BT",
                "split": "train",
                "revision": "fineweb-sha",
                "streaming": True,
            },
        )
    ]


def test_wikitext_loader_requires_raw_test_and_pinned_revision() -> None:
    spec = DatasetSpec(
        repo_id="Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        split="test",
        revision="wiki-sha",
        streaming=False,
    )
    loader = RecordingLoader()

    result = load_wikitext_test(spec, loader=loader)

    assert result == [{"id": "row", "text": "content"}]
    assert loader.calls[0] == (
        ("Salesforce/wikitext",),
        {
            "name": "wikitext-103-raw-v1",
            "split": "test",
            "revision": "wiki-sha",
            "streaming": False,
        },
    )
