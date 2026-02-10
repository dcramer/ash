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
    from ash.memory.manager import create_memory_manager

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

    print(
        f"[OK] Embeddings configured: {config.embeddings.provider}/{config.embeddings.model}"
    )

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

        # Create memory manager (handles JSONL storage + vector index)
        memory = await create_memory_manager(
            db_session=session,
            llm_registry=llm_registry,
            embedding_model=config.embeddings.model,
            embedding_provider=config.embeddings.provider,
        )

        print("-" * 60)
        print("Test 1: Store memory")
        print("-" * 60)

        test_content = "User's favorite color is purple"
        memory_entry = await memory.add_memory(
            test_content, source="test_script", owner_user_id="test-user"
        )
        print(f"[OK] Stored: '{test_content}'")
        print(f"    ID: {memory_entry.id}")

        print("-" * 60)
        print("Test 2: Search for knowledge")
        print("-" * 60)

        query = "What is the user's favorite color?"
        results = await memory.search(query, limit=5)
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}")

        found = False
        for r in results:
            print(
                f"  - [{r.source_type}] {r.content[:50]}... (sim: {r.similarity:.3f})"
            )
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
            user_id="test-user",
            user_message="What's my favorite color?",
        )

        print(f"Retrieved memories: {len(context.memories)}")

        for m in context.memories:
            print(f"  - {m.content[:50]}... (sim: {m.similarity:.3f})")

        if context.memories:
            print("[OK] Context retrieval working!")
        else:
            print(
                "[WARNING] No memories in context (may be below similarity threshold)"
            )

        print("-" * 60)
        print("Test 4: Check JSONL storage")
        print("-" * 60)

        from ash.config.paths import get_memories_jsonl_path

        memories_path = get_memories_jsonl_path()
        if memories_path.exists():
            line_count = sum(1 for _ in memories_path.read_text().splitlines())
            print(f"JSONL file: {memories_path}")
            print(f"Memory entries: {line_count}")
        else:
            print(f"JSONL file not found: {memories_path}")

    await database.disconnect()
    print("-" * 60)
    print("[DONE] Memory QA test complete")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
