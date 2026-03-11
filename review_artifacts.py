from __future__ import annotations

import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import boto3
except ImportError:  # optional dependency
    boto3 = None


@dataclass(frozen=True)
class ArtifactRef:
    backend: str
    key: str
    content_type: str = ""
    local_path: str = ""


class ArtifactStore:
    backend_name = "abstract"

    def put_file(self, key: str, src_path: str, content_type: Optional[str] = None) -> ArtifactRef:
        raise NotImplementedError

    def put_bytes(self, key: str, data: bytes, content_type: str) -> ArtifactRef:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def get_download_ref(self, key: str) -> str:
        raise NotImplementedError


class FilesystemArtifactStore(ArtifactStore):
    backend_name = "filesystem"

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        return (self.root / key).resolve()

    def put_file(self, key: str, src_path: str, content_type: Optional[str] = None) -> ArtifactRef:
        destination = self._resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, destination)
        return ArtifactRef(
            backend=self.backend_name,
            key=key,
            content_type=content_type or mimetypes.guess_type(destination.name)[0] or "",
            local_path=str(destination),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str) -> ArtifactRef:
        destination = self._resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return ArtifactRef(
            backend=self.backend_name,
            key=key,
            content_type=content_type,
            local_path=str(destination),
        )

    def get_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def get_download_ref(self, key: str) -> str:
        return str(self._resolve(key))


class MinioArtifactStore(ArtifactStore):
    backend_name = "minio"

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        region: str = "",
        use_ssl: bool = True,
        key_prefix: str = "",
    ):
        if boto3 is None:
            raise ImportError("boto3 is required for MinioArtifactStore")
        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region or None,
            use_ssl=use_ssl,
        )
        self.bucket = bucket
        self.key_prefix = key_prefix.strip().strip("/")

    def _object_key(self, key: str) -> str:
        if not self.key_prefix:
            return key
        return f"{self.key_prefix}/{key}"

    def put_file(self, key: str, src_path: str, content_type: Optional[str] = None) -> ArtifactRef:
        object_key = self._object_key(key)
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        if extra:
            self.client.upload_file(src_path, self.bucket, object_key, ExtraArgs=extra)
        else:
            self.client.upload_file(src_path, self.bucket, object_key)
        return ArtifactRef(backend=self.backend_name, key=key, content_type=content_type or "")

    def put_bytes(self, key: str, data: bytes, content_type: str) -> ArtifactRef:
        object_key = self._object_key(key)
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=data, ContentType=content_type)
        return ArtifactRef(backend=self.backend_name, key=key, content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        object_key = self._object_key(key)
        response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        object_key = self._object_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception:
            return False

    def get_download_ref(self, key: str) -> str:
        object_key = self._object_key(key)
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": object_key},
            ExpiresIn=3600,
        )


def create_artifact_store_from_config(config: dict) -> ArtifactStore:
    backend = (config.get("artifact_backend") or config.get("ARTIFACT_BACKEND") or "filesystem").strip().lower()
    if backend == "minio":
        return MinioArtifactStore(
            endpoint_url=config.get("artifact_s3_endpoint_url") or config.get("ARTIFACT_S3_ENDPOINT_URL") or "",
            access_key_id=config.get("artifact_s3_access_key_id") or config.get("ARTIFACT_S3_ACCESS_KEY_ID") or "",
            secret_access_key=config.get("artifact_s3_secret_access_key") or config.get("ARTIFACT_S3_SECRET_ACCESS_KEY") or "",
            bucket=config.get("artifact_s3_bucket") or config.get("ARTIFACT_S3_BUCKET") or "",
            region=config.get("artifact_s3_region") or config.get("ARTIFACT_S3_REGION") or "",
            use_ssl=str(config.get("artifact_s3_use_ssl") or config.get("ARTIFACT_S3_USE_SSL") or "1").strip().lower()
            not in {"0", "false", "no", "off"},
            key_prefix=config.get("artifact_s3_key_prefix") or config.get("ARTIFACT_S3_KEY_PREFIX") or "",
        )

    root = config.get("artifact_filesystem_root") or config.get("ARTIFACT_FILESYSTEM_ROOT") or os.path.join(".", "data", "artifacts")
    return FilesystemArtifactStore(Path(root))
