#!/usr/bin/env python3
"""Deterministic fake weather lookup for evals.

Usage:
  python /workspace/evals/fixtures/fake_weather.py --city "San Francisco"
"""

from __future__ import annotations

import argparse
import json

FAKE_WEATHER = {
    "san francisco": {"condition": "foggy", "temp_f": 58},
    "new york": {"condition": "clear", "temp_f": 41},
    "seattle": {"condition": "light rain", "temp_f": 49},
}


def _normalize_city(value: str) -> str:
    normalized = value.strip().casefold()
    if "," in normalized:
        normalized = normalized.split(",", 1)[0].strip()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()

    key = _normalize_city(args.city)
    weather = FAKE_WEATHER.get(key)
    if weather is None:
        print(json.dumps({"city": args.city, "error": "city_not_found"}))
        return 2

    print(
        json.dumps(
            {
                "city": args.city,
                "condition": weather["condition"],
                "temp_f": weather["temp_f"],
                "source": "fake_weather_file",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
