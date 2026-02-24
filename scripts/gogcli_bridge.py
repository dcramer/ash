#!/usr/bin/env python3
"""Wrapper entrypoint for bundled gog bridge runtime."""

from ash.skills.bundled.gog.gogcli_bridge import main

if __name__ == "__main__":
    raise SystemExit(main())
