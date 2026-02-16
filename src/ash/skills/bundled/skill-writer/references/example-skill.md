# Example: Weather Skill

A complete working skill that fetches weather data using an API.

## Directory Structure

```
workspace/skills/weather/
  SKILL.md
  scripts/weather.py
```

## SKILL.md

```markdown
---
description: Get current weather and forecast for a location
authors:
  - alice
rationale: User wants quick weather checks without leaving the chat
allowed_tools:
  - bash
env:
  - WEATHER_API_KEY
---

Get the weather for the user's requested location.

## Steps

1. Run the weather script with the location:
   ```bash
   uv run /workspace/skills/weather/scripts/weather.py "<location>"
   ```

2. Report the results in a clean format:
   - Current temperature and conditions
   - Today's high/low
   - Brief forecast if available

If the script fails, report the error message.
```

## scripts/weather.py

```python
# /// script
# dependencies = ["httpx"]
# ///
"""Fetch weather data from OpenWeatherMap API."""

import os
import sys

import httpx

API_KEY = os.environ.get("WEATHER_API_KEY")
if not API_KEY:
    print("Error: WEATHER_API_KEY not set", file=sys.stderr)
    sys.exit(1)

location = sys.argv[1] if len(sys.argv) > 1 else "San Francisco"

resp = httpx.get(
    "https://api.openweathermap.org/data/2.5/weather",
    params={"q": location, "appid": API_KEY, "units": "imperial"},
)
resp.raise_for_status()
data = resp.json()

temp = data["main"]["temp"]
desc = data["weather"][0]["description"]
high = data["main"]["temp_max"]
low = data["main"]["temp_min"]

print(f"Location: {data['name']}")
print(f"Current: {temp}F, {desc}")
print(f"High: {high}F / Low: {low}F")
```

## Config Required

```toml
# ~/.ash/config.toml
[skills.weather]
WEATHER_API_KEY = "your-openweathermap-api-key"
```

## Key Patterns

1. **Instructions are imperative** - "Run the weather script" not "You can use the script"
2. **Full paths** - `/workspace/skills/weather/scripts/weather.py`
3. **PEP 723 dependencies** - `httpx` declared inline, resolved by `uv run`
4. **Error handling** - Script checks env var, instructions say "report the error"
5. **Config via env** - API key in `env:` frontmatter, provided via `[skills.weather]` config
