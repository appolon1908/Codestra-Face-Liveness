"""Domain errors mapped to stable, machine-readable API error codes."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    UNAUTHORIZED = "UNAUTHORIZED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    BUSY = "BUSY"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    CHALLENGE_NOT_FOUND = "CHALLENGE_NOT_FOUND"
    CHALLENGE_EXPIRED = "CHALLENGE_EXPIRED"
    CHALLENGE_ALREADY_USED = "CHALLENGE_ALREADY_USED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.METHOD_NOT_ALLOWED: 405,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.INVALID_IMAGE: 422,
    ErrorCode.IMAGE_TOO_SMALL: 422,
    ErrorCode.NO_FACE: 422,
    ErrorCode.MULTIPLE_FACES: 422,
    ErrorCode.FACE_TOO_SMALL: 422,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.BUSY: 503,
    ErrorCode.DEADLINE_EXCEEDED: 503,
    ErrorCode.CHALLENGE_NOT_FOUND: 404,
    ErrorCode.CHALLENGE_EXPIRED: 410,
    ErrorCode.CHALLENGE_ALREADY_USED: 409,
    ErrorCode.INFERENCE_FAILED: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}

# Errors after which the caller may retry the identical request.
RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.BUSY,
        ErrorCode.DEADLINE_EXCEEDED,
        ErrorCode.INFERENCE_FAILED,
        ErrorCode.INTERNAL_ERROR,
    }
)


class LivenessError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status_code(self) -> int:
        return HTTP_STATUS[self.code]

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE


# Pipeline stage at which each error rejects a request (label for rejection metrics).
STAGE: dict[ErrorCode, str] = {
    ErrorCode.INVALID_REQUEST: "request",
    ErrorCode.NOT_FOUND: "routing",
    ErrorCode.METHOD_NOT_ALLOWED: "routing",
    ErrorCode.UNAUTHORIZED: "auth",
    ErrorCode.PAYLOAD_TOO_LARGE: "input",
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: "input",
    ErrorCode.INVALID_IMAGE: "input",
    ErrorCode.IMAGE_TOO_SMALL: "input",
    ErrorCode.NO_FACE: "quality",
    ErrorCode.MULTIPLE_FACES: "quality",
    ErrorCode.FACE_TOO_SMALL: "quality",
    ErrorCode.MODEL_UNAVAILABLE: "model",
    ErrorCode.BUSY: "admission",
    ErrorCode.DEADLINE_EXCEEDED: "deadline",
    ErrorCode.CHALLENGE_NOT_FOUND: "challenge",
    ErrorCode.CHALLENGE_EXPIRED: "challenge",
    ErrorCode.CHALLENGE_ALREADY_USED: "challenge",
    ErrorCode.INFERENCE_FAILED: "model",
    ErrorCode.INTERNAL_ERROR: "internal",
}
