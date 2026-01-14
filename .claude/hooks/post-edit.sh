#!/bin/bash
# Auto-format Python files after Edit/Write operations
# Reads tool info from stdin as JSON

# Parse input
INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input // empty')

# Only process Edit and Write tools
if [[ "$TOOL_NAME" != "Edit" && "$TOOL_NAME" != "Write" ]]; then
  exit 0
fi

# Extract file path from tool input
FILE_PATH=$(echo "$TOOL_INPUT" | jq -r '.file_path // empty')

# Only format Python files
if [[ "$FILE_PATH" == *.py ]]; then
  uv run ruff format "$FILE_PATH" 2>/dev/null
  uv run ruff check --fix "$FILE_PATH" 2>/dev/null
fi
