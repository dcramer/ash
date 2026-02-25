from __future__ import annotations

import pytest

from ash.rpc.methods.browser import _required_user_id


def test_required_user_id_accepts_non_empty_value() -> None:
    assert _required_user_id({"user_id": " user-1 "}) == "user-1"


def test_required_user_id_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="user_id is required"):
        _required_user_id({})


def test_required_user_id_rejects_blank_value() -> None:
    with pytest.raises(ValueError, match="user_id is required"):
        _required_user_id({"user_id": "   "})
