"""
slm_provider.py -- Small Language Model (Gemma 3 1B) provider via onnxruntime-genai.

Feature-gated: only active when MEMPALACE_NLP_SLM=1.
Model is lazily loaded on first use. Thread-safe model loading with lock.
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Prompt templates for different tasks
SENTIMENT_PROMPT = """Analyze the sentiment of the following text and respond with exactly one word: positive, negative, or neutral.

Text: {text}

Sentiment:"""

TRIPLES_PROMPT = """Extract subject-predicate-object triples from the following text.
Return a JSON array of objects with "subject", "predicate", "object" keys.
If no triples can be extracted, return an empty array [].

Text: {text}

Triples:"""

COREF_PROMPT = """Resolve pronoun references in the following text.
Return a JSON array of objects with "pronoun" and "referent" keys.
If no pronouns need resolution, return an empty array [].

Text: {text}

Coreferences:"""


class SLMProvider:
    """NLP provider using Gemma 3 1B ONNX for nuanced NLP tasks."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._og = None
        self._load_lock = threading.Lock()
        self._loaded = False
        self._available = None

    @property
    def name(self) -> str:
        return "slm"

    @property
    def capabilities(self) -> set:
        return {"sentiment", "triples", "coref"}

    def _ensure_loaded(self):
        """Lazily load model and tokenizer on first use. Thread-safe."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                import onnxruntime_genai as og

                self._og = og

                from .model_manager import ModelManager

                mm = ModelManager.get()
                model_path = mm.ensure_model("gemma-3-1b-onnx")

                if model_path is None:
                    logger.debug("Gemma model not available via ModelManager")
                    self._available = False
                else:
                    self._model = og.Model(str(model_path))
                    self._tokenizer = og.Tokenizer(self._model)
                    self._available = True
            except ImportError:
                logger.debug("onnxruntime_genai not installed — SLMProvider unavailable")
                self._available = False
            except Exception as e:
                logger.warning(f"Failed to initialize SLM: {e}")
                self._available = False
            self._loaded = True

    def is_available(self) -> bool:
        """Check if onnxruntime_genai is importable and model is available."""
        env_slm = os.environ.get("MEMPALACE_NLP_SLM")
        feature_enabled = env_slm in ("1", "true", "yes", "on")

        if not feature_enabled:
            return False

        self._ensure_loaded()
        return self._available is True

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using the SLM.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string, or empty string on failure.
        """
        if not self._available or self._model is None or self._tokenizer is None:
            return ""
        try:
            tokens = self._tokenizer.encode(prompt)
            params = self._og.GeneratorParams(self._model)
            params.set_search_options(max_length=max_tokens)
            params.input_ids = tokens
            output_tokens = self._model.generate(params)
            return self._tokenizer.decode(output_tokens[0])
        except Exception as e:
            logger.warning(f"SLM generation failed: {e}")
            return ""

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using prompted generation."""
        self._ensure_loaded()
        if not self._available:
            return "neutral"
        prompt = SENTIMENT_PROMPT.format(text=text)
        result = self.generate(prompt, max_tokens=10)
        result = result.strip().lower()
        if "positive" in result:
            return "positive"
        elif "negative" in result:
            return "negative"
        return "neutral"

    def extract_triples(self, text: str) -> List[Dict]:
        """Extract triples using prompted generation."""
        self._ensure_loaded()
        if not self._available:
            return []
        prompt = TRIPLES_PROMPT.format(text=text)
        result = self.generate(prompt, max_tokens=512)
        return self._parse_json_list(result, ["subject", "predicate", "object"])

    def resolve_coreferences(self, text: str) -> List[Dict]:
        """Resolve coreferences using prompted generation."""
        self._ensure_loaded()
        if not self._available:
            return []
        prompt = COREF_PROMPT.format(text=text)
        result = self.generate(prompt, max_tokens=256)
        return self._parse_json_list(result, ["pronoun", "referent"])

    def extract_entities(self, text: str) -> List[Dict]:
        """Not supported by SLM provider."""
        return []

    def split_sentences(self, text: str) -> List[str]:
        """Not supported by SLM provider."""
        return []

    def classify_text(self, text: str, labels: List[str]) -> Optional[Dict]:
        """Not supported by SLM provider."""
        return None

    @staticmethod
    def _parse_json_list(text: str, required_keys: list) -> List[Dict]:
        """Parse a JSON array from generated text, validating required keys."""
        try:
            # Find JSON array in the output
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1:
                return []
            data = json.loads(text[start : end + 1])
            if not isinstance(data, list):
                return []
            return [
                item
                for item in data
                if isinstance(item, dict) and all(k in item for k in required_keys)
            ]
        except (json.JSONDecodeError, ValueError):
            return []
