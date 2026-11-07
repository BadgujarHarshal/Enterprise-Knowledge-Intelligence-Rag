# tests/generation/test_llm_generator.py

import pytest

from app.generation.llm_generator import (
    BaseLLMGenerator,
    HuggingFaceLLMGenerator,
)


class FakeLLM(BaseLLMGenerator):
    def generate(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.0,
    ):
        return "Generated answer [1]."


def test_base_interface_is_abstract():
    assert BaseLLMGenerator.__abstractmethods__


def test_fake_generator_works():
    generator = FakeLLM()

    answer = generator.generate(
        "Question",
    )

    assert answer == "Generated answer [1]."


def test_model_name_required():
    with pytest.raises(ValueError):
        HuggingFaceLLMGenerator("   ")


def test_model_name_must_be_string():
    with pytest.raises(TypeError):
        HuggingFaceLLMGenerator(None)


def test_empty_prompt_rejected():
    generator = HuggingFaceLLMGenerator(
        "test-model"
    )

    with pytest.raises(ValueError):
        generator.generate("   ")


def test_invalid_max_tokens_rejected():
    generator = HuggingFaceLLMGenerator(
        "test-model"
    )

    with pytest.raises(ValueError):
        generator.generate(
            "Question",
            max_new_tokens=0,
        )


def test_negative_temperature_rejected():
    generator = HuggingFaceLLMGenerator(
        "test-model"
    )

    with pytest.raises(ValueError):
        generator.generate(
            "Question",
            temperature=-0.1,
        )


def test_lazy_loading():
    generator = HuggingFaceLLMGenerator(
        "test-model"
    )

    assert generator._pipeline is None