from __future__ import annotations

from enum import StrEnum


class FileErrorCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    INVALID_PATH = "INVALID_PATH"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    ACCESS_DENIED = "ACCESS_DENIED"
    READ_ONLY = "READ_ONLY"
    PATH_TOO_LONG = "PATH_TOO_LONG"
    INVALID_NAME = "INVALID_NAME"
    DIRECTORY_NOT_EMPTY = "DIRECTORY_NOT_EMPTY"
    IN_USE = "IN_USE"
    CROSS_DEVICE_FAILURE = "CROSS_DEVICE_FAILURE"
    IO_ERROR = "IO_ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class OverwritePolicy(StrEnum):
    FAIL_IF_EXISTS = "fail_if_exists"
    REPLACE = "replace"
    RENAME_NEW = "rename_new"


class FileOperationError(RuntimeError):
    def __init__(self, code: FileErrorCode, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
