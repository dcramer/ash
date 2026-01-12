"""Embedding generation for semantic search."""

from ash.llm import LLMRegistry


class EmbeddingGenerator:
    """Generate embeddings for text using LLM providers."""

    def __init__(
        self,
        registry: LLMRegistry,
        model: str | None = None,
        provider: str = "openai",
    ):
        """Initialize embedding generator.

        Args:
            registry: LLM provider registry.
            model: Embedding model to use.
            provider: Provider name (default: openai, as Anthropic doesn't support embeddings).
        """
        self._registry = registry
        self._model = model
        self._provider_name = provider

    @property
    def _provider(self):
        """Get the embedding provider."""
        return self._registry.get(self._provider_name)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions.

        Note: text-embedding-3-small produces 1536-dimensional vectors.
        """
        return 1536

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors (1:1 correspondence with input).

        Raises:
            ValueError: If any text is empty or whitespace-only.
        """
        if not texts:
            return []

        # Validate no empty strings - OpenAI API rejects them
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Empty or whitespace-only text at index {i}")

        return await self._provider.embed(texts, model=self._model)

    async def embed_with_chunking(
        self,
        text: str,
        chunk_size: int = 8000,
        overlap: int = 200,
    ) -> list[tuple[str, list[float]]]:
        """Embed long text by chunking.

        Args:
            text: Text to embed.
            chunk_size: Maximum characters per chunk.
            overlap: Overlap between chunks.

        Returns:
            List of (chunk_text, embedding) tuples.
        """
        chunks = self._chunk_text(text, chunk_size, overlap)
        embeddings = await self.embed_batch(chunks)
        return list(zip(chunks, embeddings, strict=True))

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk.
            chunk_size: Maximum characters per chunk.
            overlap: Overlap between chunks.

        Returns:
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation
                for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n", "\n\n"]:
                    pos = text.rfind(sep, start + chunk_size // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]  # Filter empty chunks
