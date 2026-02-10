"""User profile storage operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import UserProfile


async def get_or_create_user_profile(
    session: AsyncSession,
    user_id: str,
    provider: str,
    username: str | None = None,
    display_name: str | None = None,
) -> UserProfile:
    """Get or create user profile.

    Args:
        session: Database session.
        user_id: User ID.
        provider: Provider name (e.g., "telegram").
        username: Optional username.
        display_name: Optional display name.

    Returns:
        User profile (existing or newly created).
    """
    result = await session.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        profile = UserProfile(
            user_id=user_id,
            provider=provider,
            username=username,
            display_name=display_name,
        )
        session.add(profile)
    else:
        if username and profile.username != username:
            profile.username = username
        if display_name and profile.display_name != display_name:
            profile.display_name = display_name

    await session.flush()
    return profile
