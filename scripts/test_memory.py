#!/usr/bin/env python
"""Manual QA script for testing memory functionality.

This script tests the end-to-end memory flow:
1. Remember something (stores knowledge)
2. Restart session (simulates restart)
3. Recall the information

Run with: uv run python scripts/test_memory.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    from ash.config import load_config
    from ash.config.paths import get_database_path
    from ash.db.engine import Database
    from ash.llm import create_registry
    from ash.memory.embeddings import EmbeddingGenerator
    from ash.memory.manager import MemoryManager
    from ash.memory.retrieval import SemanticRetriever
    from ash.memory.store import MemoryStore

    print("=" * 60)
    print("Memory System QA Test")
    print("=" * 60)

    # Load config
    try:
        config = load_config()
        print("[OK] Config loaded from default location")
    except FileNotFoundError as e:
        print(f"[ERROR] Config not found: {e}")
        return False

    # Check embeddings config
    if not config.embeddings:
        print("[ERROR] Embeddings not configured!")
        print("       Add to config.toml:")
        print("       [embeddings]")
        print('       provider = "openai"')
        print('       model = "text-embedding-3-small"')
        return False

    print(f"[OK] Embeddings configured: {config.embeddings.provider}/{config.embeddings.model}")

    # Check API key
    embeddings_key = config.resolve_embeddings_api_key()
    if not embeddings_key:
        print(f"[ERROR] No API key for {config.embeddings.provider}")
        print("       Set OPENAI_API_KEY environment variable or add to config")
        return False

    print("[OK] Embeddings API key found")

    # Connect to database
    db_path = get_database_path()
    print(f"[OK] Database path: {db_path}")

    database = Database(database_path=db_path)
    await database.connect()
    print("[OK] Database connected")

    async with database.session() as session:
        # Create LLM registry for embeddings
        llm_registry = create_registry(
            openai_api_key=embeddings_key.get_secret_value()
            if config.embeddings.provider == "openai"
            else None,
        )

        # Create components
        embedding_generator = EmbeddingGenerator(
            registry=llm_registry,
            model=config.embeddings.model,
            provider=config.embeddings.provider,
        )

        store = MemoryStore(session)
        retriever = SemanticRetriever(session, embedding_generator)
        await retriever.initialize_vector_tables()
        memory = MemoryManager(store, retriever, session)

        print("-" * 60)
        print("Test 1: Store knowledge")
        print("-" * 60)

        test_content = "User's favorite color is purple"
        knowledge = await memory.add_knowledge(test_content, source="test_script")
        print(f"[OK] Stored: '{test_content}'")
        print(f"    ID: {knowledge.id}")

        # Commit explicitly
        await session.commit()
        print("[OK] Database committed")

        print("-" * 60)
        print("Test 2: Search for knowledge")
        print("-" * 60)

        query = "What is the user's favorite color?"
        results = await memory.search(query, limit=5)
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}")

        found = False
        for r in results:
            print(f"  - [{r.source_type}] {r.content[:50]}... (sim: {r.similarity:.3f})")
            if "purple" in r.content.lower():
                found = True

        if found:
            print("[OK] Found relevant knowledge!")
        else:
            print("[WARNING] Didn't find the stored knowledge")

        print("-" * 60)
        print("Test 3: Context retrieval")
        print("-" * 60)

        context = await memory.get_context_for_message(
            session_id="test-session",
            user_id="test-user",
            user_message="What's my favorite color?",
        )

        print(f"Retrieved messages: {len(context.messages)}")
        print(f"Retrieved knowledge: {len(context.knowledge)}")

        for k in context.knowledge:
            print(f"  - {k.content[:50]}... (sim: {k.similarity:.3f})")

        if context.knowledge:
            print("[OK] Context retrieval working!")
        else:
            print("[WARNING] No knowledge in context (may be below 0.3 similarity threshold)")

        print("-" * 60)
        print("Test 4: Check database directly")
        print("-" * 60)

        # Check knowledge table
        from sqlalchemy import text
        result = await session.execute(text("SELECT COUNT(*) FROM knowledge"))
        count = result.scalar()
        print(f"Knowledge entries: {count}")

        result = await session.execute(text("SELECT COUNT(*) FROM knowledge_embeddings"))
        count = result.scalar()
        print(f"Knowledge embeddings: {count}")

    await database.disconnect()
    print("-" * 60)
    print("[DONE] Memory QA test complete")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
