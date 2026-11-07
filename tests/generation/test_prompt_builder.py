# tests/generation/test_prompt_builder.py

from dataclasses import dataclass

import pytest

from app.generation.prompt_builder import (
    PromptBuilder,
)


@dataclass
class FakeDocument:
    chunk_id: str
    title: str
    content: str


@dataclass
class FakeResult:
    rank: int
    score: float
    document: FakeDocument
    hop: int


def make_result(
    chunk_id="chunk_1",
    title="Document A",
    content="Document A contains the required fact.",
    score=0.9,
    rank=1,
    hop=1,
):
    return FakeResult(
        rank=rank,
        score=score,
        document=FakeDocument(
            chunk_id=chunk_id,
            title=title,
            content=content,
        ),
        hop=hop,
    )


def test_build_context_numbers_evidence():
    builder = PromptBuilder()

    results = [
        make_result(chunk_id="chunk_1"),
        make_result(
            chunk_id="chunk_2",
            title="Document B",
        ),
    ]

    contexts = builder.build_context(results)

    assert len(contexts) == 2
    assert contexts[0].citation_id == 1
    assert contexts[1].citation_id == 2


def test_build_context_preserves_metadata():
    builder = PromptBuilder()

    result = make_result(
        score=0.73,
        hop=2,
    )

    contexts = builder.build_context([result])

    assert contexts[0].score == 0.73
    assert contexts[0].hop == 2
    assert contexts[0].chunk_id == "chunk_1"


def test_build_prompt_contains_question():
    builder = PromptBuilder()

    prompt, _ = builder.build_prompt(
        "What happened?",
        [make_result()],
    )

    assert "What happened?" in prompt


def test_build_prompt_contains_evidence():
    builder = PromptBuilder()

    prompt, _ = builder.build_prompt(
        "What happened?",
        [
            make_result(
                content="The event happened in 1950."
            )
        ],
    )

    assert "The event happened in 1950." in prompt


def test_build_prompt_contains_citation_instruction():
    builder = PromptBuilder()

    prompt, _ = builder.build_prompt(
        "What happened?",
        [make_result()],
    )

    assert "[N]" in prompt
    assert "Every factual claim" in prompt


def test_context_limit():
    builder = PromptBuilder(
        max_context_chunks=2,
    )

    results = [
        make_result(chunk_id="1"),
        make_result(chunk_id="2"),
        make_result(chunk_id="3"),
    ]

    contexts = builder.build_context(results)

    assert len(contexts) == 2


def test_content_limit():
    builder = PromptBuilder(
        max_content_chars=10,
    )

    result = make_result(
        content="12345678901234567890",
    )

    contexts = builder.build_context([result])

    assert contexts[0].content == "1234567890"


def test_empty_results_rejected():
    builder = PromptBuilder()

    with pytest.raises(ValueError):
        builder.build_prompt(
            "Question",
            [],
        )


def test_empty_question_rejected():
    builder = PromptBuilder()

    with pytest.raises(ValueError):
        builder.build_prompt(
            "   ",
            [make_result()],
        )


def test_missing_content_rejected():
    builder = PromptBuilder()

    result = {
        "chunk_id": "chunk_1",
        "title": "Document",
        "content": "",
    }

    with pytest.raises(ValueError):
        builder.build_context([result])


def test_missing_chunk_id_rejected():
    builder = PromptBuilder()

    result = {
        "title": "Document",
        "content": "Some evidence.",
    }

    with pytest.raises(ValueError):
        builder.build_context([result])


def test_dictionary_results_supported():
    builder = PromptBuilder()

    result = {
        "rank": 1,
        "score": 0.8,
        "hop": 2,
        "document": {
            "chunk_id": "chunk_1",
            "title": "Document",
            "content": "Evidence.",
        },
    }

    contexts = builder.build_context([result])

    assert len(contexts) == 1
    assert contexts[0].title == "Document"
    assert contexts[0].content == "Evidence."
    assert contexts[0].hop == 2