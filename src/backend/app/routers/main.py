from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint health check.

    Returns:
        dict: Simple acknowledgment message
    """
    return {"status": "ok", "message": "API is running"}
