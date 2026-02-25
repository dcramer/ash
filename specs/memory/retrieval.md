# Memory Retrieval

## Retrieval Pipeline

Automatic context injection uses a multi-stage retrieval pipeline:

| Stage | Input | Method | Score |
|-------|-------|--------|-------|
| 0. Query planning | User message + chat metadata + recent chat history | Fast LLM rewrites retrieval query and proposes supplemental lookups for missing context | N/A |
| 1. Primary search | Planned query | Hybrid vector + person-graph search | Vector similarity (0-1) or 0.75/0.55 for graph |
| 2. Cross-context | Participant person IDs | ABOUT edges across all owners | 0.7 fixed |
| 3. Multi-hop BFS | Seed persons from stages 1-2 | 2-hop graph traversal | 0.5 (1-hop) / 0.3 (2-hop) |
| 4. RRF fusion | All stage results | Reciprocal Rank Fusion (K=60) | Combined RRF score |

Stage 2 excludes the querying user's own memories (covered by stage 1). Stage 3 skips SUPERSEDES edges and filters by portable/privacy. RRF fusion boosts memories appearing in multiple stages.

## Provenance Invariant

Active memories must have `LEARNED_IN` provenance. Memories missing `LEARNED_IN`
are invalid and must be excluded from disclosure/retrieval paths until
remediated (backfilled or archived).

Query planning is optional and bounded by config (`memory.query_planning_*`). The retrieval call can fetch more memories than final prompt budget, then prune to `memory.context_injection_limit`.

Planner output has two parts:
- `query` (primary retrieval query)
- `lookup_queries` (supplemental contextual lookups)

Both are executed for retrieval, deduplicated by memory ID, and merged before final pruning.
