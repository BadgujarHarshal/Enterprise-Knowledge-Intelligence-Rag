# tests/generation/test_answer_generator.py

from dataclasses import dataclass

import pytest

from app.generation.answer_generator import (
    AnswerGenerator,
)
from app.generation.llm_generator import (
    BaseLLMGenerator,
)
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


class FakeLLM(BaseLLMGenerator):
    def __init__(self, answer):
        self.answer = answer
        self.last_prompt = None
        self.last_max_new_tokens = None
        self.last_temperature = None

    def generate(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.0,
    ):
        self.last_prompt = prompt
        self.last_max_new_tokens = max_new_tokens
        self.last_temperature = temperature
        return self.answer


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


def test_generate_returns_answer():
    llm = FakeLLM(
        "The answer is 1950 [1]."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    result = generator.generate(
        "When did it happen?",
        [make_result()],
    )

    assert result.answer == "The answer is 1950 [1]."
    assert result.question == "When did it happen?"


def test_citations_are_extracted():
    llm = FakeLLM(
        "The answer is supported by [1] and [2]."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    results = [
        make_result(chunk_id="chunk_1"),
        make_result(
            chunk_id="chunk_2",
            title="Document B",
        ),
    ]

    result = generator.generate(
        "Question",
        results,
    )

    assert result.citations == (1, 2)


def test_duplicate_citations_are_removed():
    llm = FakeLLM(
        "The answer [1] is supported by [1]."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    result = generator.generate(
        "Question",
        [make_result()],
    )

    assert result.citations == (1,)


def test_invalid_citations_are_ignored():
    llm = FakeLLM(
        "The answer [1] has an invalid citation [99]."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    result = generator.generate(
        "Question",
        [make_result()],
    )

    assert result.citations == (1,)


def test_no_citations_returns_empty_tuple():
    llm = FakeLLM(
        "The answer contains no citation."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    result = generator.generate(
        "Question",
        [make_result()],
    )

    assert result.citations == ()


def test_prompt_is_forwarded_to_llm():
    llm = FakeLLM(
        "Answer [1]."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    generator.generate(
        "What happened?",
        [make_result()],
    )

    assert llm.last_prompt is not None
    assert "What happened?" in llm.last_prompt
    assert "Document A" in llm.last_prompt


def test_generation_parameters_are_forwarded():
    llm = FakeLLM(
        "Answer [1]."
    )

    generator = AnswerGenerator(
        llm=llm,
        max_new_tokens=128,
        temperature=0.2,
    )

    generator.generate(
        "Question",
        [make_result()],
    )

    assert llm.last_max_new_tokens == 128
    assert llm.last_temperature == 0.2


def test_empty_question_rejected():
    llm = FakeLLM(
        "Answer [1]."
    )

    generator = AnswerGenerator(
        llm=llm,
    )

    with pytest.raises(ValueError):
        generator.generate(
            "   ",
            [make_result()],
        )


def test_empty_llm_answer_rejected():
    llm = FakeLLM("")

    generator = AnswerGenerator(
        llm=llm,
    )

    with pytest.raises(RuntimeError):
        generator.generate(
            "Question",
            [make_result()],
        )


def test_prompt_context_is_returned():
    llm = FakeLLM(
        "Answer [1]."
    )

    generator = AnswerGenerator(
        llm=llm,
        prompt_builder=PromptBuilder(
            max_context_chunks=1
        ),
    )

    result = generator.generate(
        "Question",
        [
            make_result(chunk_id="chunk_1"),
            make_result(chunk_id="chunk_2"),
        ],
    )

    assert len(result.citation_context) == 1
    assert result.citation_context[0].citation_id == 1


def test_custom_llm_can_be_used():
    class CustomLLM(BaseLLMGenerator):
        def generate(
            self,
            prompt,
            max_new_tokens=256,
            temperature=0.0,
        ):
            return "Custom response [1]."

    generator = AnswerGenerator(
        llm=CustomLLM(),
    )

    result = generator.generate(
        "Question",
        [make_result()],
    )

    assert result.answer == "Custom response [1]."


def test_none_llm_rejected():
    with pytest.raises(ValueError):
        AnswerGenerator(
            llm=None,
        )