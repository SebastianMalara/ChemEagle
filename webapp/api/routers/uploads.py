from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, UploadFile

from ..runtime import persist_uploaded_files
from ..schemas import UploadResponse

router = APIRouter(tags=["uploads"])


@router.post("/uploads", response_model=UploadResponse)
def upload_files(files: List[UploadFile] = File(...), prefix: str = "web") -> UploadResponse:
    stored_paths = persist_uploaded_files(files, prefix=prefix)
    return UploadResponse(stored_paths=stored_paths)
