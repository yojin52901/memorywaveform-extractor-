"""Configurable local vision-language model providers."""

from memorywaveform_extractor.providers.ollama import OllamaVisionProvider
from memorywaveform_extractor.providers.openai_compatible import OpenAICompatibleVisionProvider

__all__ = ["OllamaVisionProvider", "OpenAICompatibleVisionProvider"]
