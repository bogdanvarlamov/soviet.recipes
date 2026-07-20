"""LLM-based extraction engine implementation.

Talks to any OpenAI-compatible chat/completions endpoint that supports vision
input. Primary target is a local llama.cpp server (llama-server) running a
Qwen3-VL model, e.g.:

    llama-server -m Qwen3-VL-*.gguf --mmproj mmproj-*.gguf \
        --host 0.0.0.0 --port 8080

which exposes an OpenAI-compatible API at http://localhost:8080/v1
"""

import base64
import io
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from openai import OpenAI

from engines.base import ExtractionEngine
from config.settings import LLMConfig
from exceptions import ExtractionError, ConfigurationError, PageSkipped


# Tool the model can call to declare that a page has no transcribable text.
# Exposing this as a real function-calling tool lets the model make the
# decision itself, so pages that are pure images/blanks are skipped up front
# instead of burning retries.
SKIP_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "skip_page",
        "description": (
            "Call this when the page contains NO transcribable text at all - "
            "for example a full-page photograph, illustration, decorative "
            "artwork, or a blank page. Use this instead of returning an empty "
            "or guessed transcription so the page is skipped without wasting "
            "retries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Brief reason the page has no transcribable text "
                        "(e.g. 'full-page photograph', 'blank page')."
                    ),
                }
            },
            "required": ["reason"],
        },
    },
}


class LLMEngine(ExtractionEngine):
    """Text extraction using a local or remote vision LLM (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM engine.

        Args:
            config: LLM-specific configuration
        """
        self.config = config
        self.model_name = config.model_name
        self.logger = logging.getLogger(__name__)
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """
        Lazily construct the OpenAI-compatible client.

        Returns:
            Configured OpenAI client instance

        Raises:
            ConfigurationError: If the client cannot be constructed
        """
        if self._client is not None:
            return self._client

        try:
            # llama.cpp does not require a real key, but the SDK insists on one.
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key or "sk-no-key-required",
                timeout=self.config.timeout,
            )
            self.logger.info(
                f"Initialized LLM client (base_url={self.config.base_url}, "
                f"model={self.model_name})"
            )
            return self._client
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize LLM client: {e}"
            ) from e

    def _encode_image(self, image_file: Path) -> str:
        """
        Read an image and return it as a base64 data URL, optionally
        downscaling it (config.max_image_size) to reduce image tokens.

        Args:
            image_file: Path to the image file

        Returns:
            A data URL string (data:<mime>;base64,<data>)
        """
        max_size = self.config.max_image_size

        if max_size:
            # Downscale (preserving aspect ratio) if the image is larger than
            # the configured longest edge, then re-encode as JPEG.
            from PIL import Image as PILImage

            with PILImage.open(image_file) as img:
                img = img.convert("RGB")
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size))
                    self.logger.debug(
                        f"Downscaled {image_file.name} to {img.size} "
                        f"(max_image_size={max_size})"
                    )
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                data = buffer.getvalue()
            mime_type = "image/jpeg"
        else:
            data = image_file.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(image_file))
            if mime_type is None:
                mime_type = "image/jpeg"

        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file using a vision LLM.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text as a string

        Raises:
            ExtractionError: If extraction fails
            PageSkipped: If the model decides the page has no transcribable text
        """
        image_file = Path(image_path)

        if not image_file.exists():
            raise ExtractionError(f"Image file not found: {image_path}")
        if not image_file.is_file():
            raise ExtractionError(f"Path is not a file: {image_path}")

        self.logger.info(f"Extracting text from: {image_file.name}")

        try:
            client = self._get_client()
            data_url = self._encode_image(image_file)

            # Offer the skip_page tool so the model can bail out of pages that
            # have no transcribable text instead of forcing a (retried) answer.
            extra_kwargs = {}
            if self.config.allow_skip:
                extra_kwargs["tools"] = [SKIP_PAGE_TOOL]
                extra_kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(
                model=self.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                **extra_kwargs,
            )

            if not response.choices:
                raise ExtractionError(
                    f"LLM returned no choices for {image_file.name}"
                )

            message = response.choices[0].message

            # If the model chose to skip the page, surface that as a PageSkipped
            # signal (terminal, non-retryable) rather than extracted text.
            if self.config.allow_skip:
                reason = self._skip_reason(message)
                if reason is not None:
                    self.logger.info(
                        f"Model skipped {image_file.name}: {reason}"
                    )
                    raise PageSkipped(reason)

            content = message.content
            if content is None:
                raise ExtractionError(
                    f"LLM returned empty content for {image_file.name}"
                )

            extracted_text = content.strip()
            self.logger.info(
                f"Successfully extracted {len(extracted_text)} characters "
                f"from {image_file.name}"
            )
            return extracted_text

        except (ExtractionError, PageSkipped):
            raise
        except Exception as e:
            error_msg = f"Failed to extract text from {image_file.name}: {e}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e

    @staticmethod
    def _skip_reason(message) -> Optional[str]:
        """
        Return the skip reason if the model called the skip_page tool, else None.

        Handles both a proper ``tool_calls`` array and the occasional model that
        emits the call under ``function_call``.

        Args:
            message: The assistant message from the chat completion response

        Returns:
            The reason string if skip_page was called, otherwise None
        """
        def parse_args(raw_args) -> str:
            if not raw_args:
                return "No transcribable text on page"
            if isinstance(raw_args, dict):
                return raw_args.get("reason", "No transcribable text on page")
            try:
                parsed = json.loads(raw_args)
                return parsed.get("reason", "No transcribable text on page")
            except (json.JSONDecodeError, AttributeError):
                return "No transcribable text on page"

        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            fn = getattr(call, "function", None)
            if fn is not None and getattr(fn, "name", None) == "skip_page":
                return parse_args(getattr(fn, "arguments", None))

        # Fallback for the legacy single function_call field.
        function_call = getattr(message, "function_call", None)
        if function_call is not None and getattr(function_call, "name", None) == "skip_page":
            return parse_args(getattr(function_call, "arguments", None))

        return None

    def validate_config(self) -> bool:
        """
        Validate that the engine is properly configured.

        Verifies that a base_url and model are set and that the server is
        reachable via the OpenAI-compatible /models endpoint.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.logger.info("Validating LLM engine configuration")

        if not self.config.base_url:
            raise ConfigurationError(
                "LLMConfig.base_url is required (e.g. http://localhost:8080/v1)"
            )
        if not self.model_name:
            raise ConfigurationError("LLMConfig.model_name is required")

        try:
            client = self._get_client()
            # Reachability check against the OpenAI-compatible server.
            client.models.list()
            self.logger.info("LLM engine configuration is valid")
            return True
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Could not reach LLM server at {self.config.base_url}: {e}. "
                f"Is llama-server running?"
            ) from e
