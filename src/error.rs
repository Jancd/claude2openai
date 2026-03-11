use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("Invalid request: {0}")]
    Validation(String),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Permission denied: {0}")]
    Permission(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Request too large: {0}")]
    OverLimit(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Host not allowed: {0}")]
    SsrfBlocked(String),

    #[error("Upstream error: {status} - {message}")]
    Upstream {
        status: u16,
        message: String,
        error_type: String,
    },

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl ProxyError {
    fn status_code(&self) -> StatusCode {
        match self {
            ProxyError::Validation(_) => StatusCode::BAD_REQUEST,
            ProxyError::Authentication(_) => StatusCode::UNAUTHORIZED,
            ProxyError::Permission(_) => StatusCode::FORBIDDEN,
            ProxyError::RateLimit(_) => StatusCode::TOO_MANY_REQUESTS,
            ProxyError::OverLimit(_) => StatusCode::PAYLOAD_TOO_LARGE,
            ProxyError::Processing(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::SsrfBlocked(_) => StatusCode::FORBIDDEN,
            ProxyError::Upstream { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
            }
            ProxyError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &str {
        match self {
            ProxyError::Validation(_) => "invalid_request_error",
            ProxyError::Authentication(_) => "authentication_error",
            ProxyError::Permission(_) => "permission_error",
            ProxyError::RateLimit(_) => "rate_limit_error",
            ProxyError::OverLimit(_) => "over_limit_error",
            ProxyError::Processing(_) => "processing_error",
            ProxyError::SsrfBlocked(_) => "permission_error",
            ProxyError::Upstream { error_type, .. } => error_type.as_str(),
            ProxyError::Internal(_) => "processing_error",
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = self.error_type().to_string();
        let message = self.to_string();

        let body = json!({
            "type": error_type,
            "error": {
                "type": error_type,
                "message": message
            }
        });

        (status, axum::Json(body)).into_response()
    }
}

/// Map upstream HTTP status to Claude error type string.
pub fn map_upstream_status_to_error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        429 => "rate_limit_error",
        413 => "over_limit_error",
        500 | 502 | 503 | 504 => "processing_error",
        _ => "processing_error",
    }
}

/// Build a ProxyError from an upstream HTTP response status.
pub fn handle_target_api_error(status: u16, target_name: &str) -> ProxyError {
    let error_type = map_upstream_status_to_error_type(status);
    let message = match status {
        400 => format!("Invalid request to {target_name}"),
        401 => format!("Authentication failed for {target_name}"),
        403 => format!("Insufficient permissions for {target_name}"),
        429 => format!("Rate limit exceeded for {target_name}"),
        413 => format!("Request exceeds limits for {target_name}"),
        500 | 502 | 503 | 504 => format!("Service error from {target_name}"),
        _ => format!("Target API ({target_name}) returned error: {status}"),
    };

    ProxyError::Upstream {
        status,
        message,
        error_type: error_type.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_upstream_status_400() {
        assert_eq!(map_upstream_status_to_error_type(400), "invalid_request_error");
    }

    #[test]
    fn test_map_upstream_status_401() {
        assert_eq!(map_upstream_status_to_error_type(401), "authentication_error");
    }

    #[test]
    fn test_map_upstream_status_403() {
        assert_eq!(map_upstream_status_to_error_type(403), "permission_error");
    }

    #[test]
    fn test_map_upstream_status_429() {
        assert_eq!(map_upstream_status_to_error_type(429), "rate_limit_error");
    }

    #[test]
    fn test_map_upstream_status_413() {
        assert_eq!(map_upstream_status_to_error_type(413), "over_limit_error");
    }

    #[test]
    fn test_map_upstream_status_5xx() {
        assert_eq!(map_upstream_status_to_error_type(500), "processing_error");
        assert_eq!(map_upstream_status_to_error_type(502), "processing_error");
        assert_eq!(map_upstream_status_to_error_type(503), "processing_error");
        assert_eq!(map_upstream_status_to_error_type(504), "processing_error");
    }

    #[test]
    fn test_map_upstream_status_unknown() {
        assert_eq!(map_upstream_status_to_error_type(418), "processing_error");
    }

    #[test]
    fn test_handle_target_api_error_400() {
        let err = handle_target_api_error(400, "test-api");
        match err {
            ProxyError::Upstream { status, message, error_type } => {
                assert_eq!(status, 400);
                assert!(message.contains("test-api"));
                assert_eq!(error_type, "invalid_request_error");
            }
            _ => panic!("Expected Upstream error"),
        }
    }

    #[test]
    fn test_handle_target_api_error_unknown_status() {
        let err = handle_target_api_error(418, "test-api");
        match err {
            ProxyError::Upstream { status, message, .. } => {
                assert_eq!(status, 418);
                assert!(message.contains("418"));
            }
            _ => panic!("Expected Upstream error"),
        }
    }

    #[test]
    fn test_proxy_error_status_codes() {
        assert_eq!(
            ProxyError::Validation("test".into()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ProxyError::Authentication("test".into()).status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ProxyError::Permission("test".into()).status_code(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            ProxyError::RateLimit("test".into()).status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ProxyError::OverLimit("test".into()).status_code(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            ProxyError::Processing("test".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            ProxyError::SsrfBlocked("test".into()).status_code(),
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn test_proxy_error_error_types() {
        assert_eq!(
            ProxyError::Validation("test".into()).error_type(),
            "invalid_request_error"
        );
        assert_eq!(
            ProxyError::Authentication("test".into()).error_type(),
            "authentication_error"
        );
        assert_eq!(
            ProxyError::SsrfBlocked("test".into()).error_type(),
            "permission_error"
        );
    }

    #[test]
    fn test_proxy_error_into_response() {
        let err = ProxyError::Validation("bad input".into());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
