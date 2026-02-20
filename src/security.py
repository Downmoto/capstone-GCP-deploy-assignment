import os

from fastapi import Header, HTTPException, status


API_KEY_HEADER = "x-api-key"


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="api key is not configured on server",
        )

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
