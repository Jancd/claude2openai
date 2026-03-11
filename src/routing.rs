use std::collections::HashMap;

use axum::http::HeaderMap;

use crate::beta_features::validate_beta_features;
use crate::error::ProxyError;

/// Default allowed hosts for SSRF protection.
const DEFAULT_ALLOWED_HOSTS: &[&str] = &["127.0.0.1", "localhost"];

/// Common path segments that are NOT model IDs.
const COMMON_PATH_SEGMENTS: &[&str] = &[
    "v1",
    "v2",
    "models",
    "messages",
    "completions",
    "chat",
    "openai",
    "api",
];

#[derive(Debug, Clone)]
pub struct TargetConfig {
    pub target_url: String,
    pub target_path_prefix: String,
}

#[derive(Debug, Clone)]
pub struct ParsedRoute {
    pub target_config: TargetConfig,
    pub claude_endpoint: String,
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HandlerType {
    Models,
    Messages,
    TokenCounting,
}

/// Parse dynamic routing URL.
///
/// Format: `/{protocol}/{host}/{path_prefix}/{model_id?}/{claude_endpoint}`
pub fn parse_dynamic_route(path: &str) -> Result<ParsedRoute, ProxyError> {
    let path = path.strip_prefix('/').unwrap_or(path);
    let parts: Vec<&str> = path.split('/').collect();

    if parts.len() < 4 {
        return Err(ProxyError::Validation(format!(
            "Invalid URL format: /{path}. Expected format: /{{protocol}}/{{host}}/{{path}}/{{endpoint}}"
        )));
    }

    let protocol = parts[0];
    if protocol != "http" && protocol != "https" {
        return Err(ProxyError::Validation(format!(
            "Invalid protocol: {protocol}. Must be 'http' or 'https'"
        )));
    }

    let host = parts[1];

    // Find Claude endpoint from the end
    let mut claude_endpoint_start: Option<usize> = None;

    for i in (2..parts.len()).rev() {
        if parts[i] == "v1" {
            let next = parts.get(i + 1).copied();
            if next == Some("models") || next == Some("messages") {
                claude_endpoint_start = Some(i);
                break;
            }
        }
    }

    let endpoint_start = claude_endpoint_start.ok_or_else(|| {
        ProxyError::Validation(format!("Could not locate Claude endpoint in URL: /{path}"))
    })?;

    let claude_endpoint = parts[endpoint_start..].join("/");
    let target_path_end = endpoint_start;

    // Determine if there's a model ID between target path and Claude endpoint
    let mut model_id: Option<String> = None;
    let mut actual_path_end = target_path_end;

    // Check if the last part before the endpoint is a model ID
    if target_path_end > 2 {
        let last_part = parts[target_path_end - 1];
        if !COMMON_PATH_SEGMENTS.contains(&last_part) && !last_part.is_empty() {
            model_id = Some(last_part.to_string());
            actual_path_end = target_path_end - 1;
        }
    }

    let target_path_prefix = if actual_path_end > 2 {
        format!("/{}", parts[2..actual_path_end].join("/"))
    } else {
        String::new()
    };

    Ok(ParsedRoute {
        target_config: TargetConfig {
            target_url: format!("{protocol}://{host}"),
            target_path_prefix,
        },
        claude_endpoint,
        model_id,
    })
}

/// Build target URL from config, endpoint, and optional model ID.
pub fn build_target_url(config: &TargetConfig, endpoint: &str, model_id: Option<&str>) -> String {
    let mut url = format!("{}{}", config.target_url, config.target_path_prefix);
    if let Some(mid) = model_id {
        url.push('/');
        url.push_str(mid);
    }
    url.push('/');
    url.push_str(endpoint);
    url
}

/// Determine handler type based on Claude endpoint string.
pub fn get_handler_type(claude_endpoint: &str) -> Result<HandlerType, ProxyError> {
    if claude_endpoint == "v1/models" {
        return Ok(HandlerType::Models);
    }
    if claude_endpoint == "v1/messages/count_tokens" {
        return Ok(HandlerType::TokenCounting);
    }
    if claude_endpoint.starts_with("v1/messages") {
        return Ok(HandlerType::Messages);
    }
    Err(ProxyError::Validation(format!(
        "Unknown Claude endpoint: {claude_endpoint}"
    )))
}

/// Check if a host is allowed based on the allowed hosts list.
pub fn is_host_allowed(host: &str, allowed_hosts: &[String]) -> bool {
    let hosts = if allowed_hosts.is_empty() {
        DEFAULT_ALLOWED_HOSTS
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    } else {
        allowed_hosts.to_vec()
    };

    let normalized = host.to_lowercase();

    hosts.iter().any(|allowed| {
        let allowed_lower = allowed.to_lowercase();
        // Exact match
        if allowed_lower == normalized {
            return true;
        }
        // Wildcard domain match (e.g., "*.example.com")
        if let Some(domain) = allowed_lower.strip_prefix("*.") {
            return normalized.ends_with(domain);
        }
        false
    })
}

/// Extract authentication headers from the incoming request.
pub fn extract_auth_headers(headers: &HeaderMap) -> HashMap<String, String> {
    let mut auth_headers = HashMap::new();

    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let api_key_header = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // If x-api-key is provided but Authorization is missing, convert it
    if let Some(api_key) = &api_key_header {
        if auth_header.is_none() {
            if api_key.starts_with("Bearer ") {
                auth_headers.insert("Authorization".to_string(), api_key.clone());
            } else {
                auth_headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
            }
        }
    }

    if let Some(auth) = auth_header {
        auth_headers.insert("Authorization".to_string(), auth);
    }

    // Forward beta feature headers
    if let Some(beta_header) = headers.get("anthropic-beta").and_then(|v| v.to_str().ok()) {
        let validated = validate_beta_features(beta_header);
        if let Some(features) = validated {
            auth_headers.insert(
                "anthropic-beta".to_string(),
                serde_json::to_string(&features).unwrap_or_else(|_| beta_header.to_string()),
            );
        } else {
            auth_headers.insert("anthropic-beta".to_string(), beta_header.to_string());
        }
    }

    auth_headers
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_dynamic_route ---

    #[test]
    fn test_parse_dynamic_route_basic() {
        let route = parse_dynamic_route("/https/api.example.com/v1/v1/messages").unwrap();
        assert_eq!(route.target_config.target_url, "https://api.example.com");
        assert_eq!(route.claude_endpoint, "v1/messages");
        assert!(route.model_id.is_none());
    }

    #[test]
    fn test_parse_dynamic_route_with_model_id() {
        let route =
            parse_dynamic_route("/https/api.example.com/v1/gpt-4/v1/messages").unwrap();
        assert_eq!(route.target_config.target_url, "https://api.example.com");
        assert_eq!(route.model_id, Some("gpt-4".to_string()));
        assert_eq!(route.claude_endpoint, "v1/messages");
    }

    #[test]
    fn test_parse_dynamic_route_models_endpoint() {
        let route = parse_dynamic_route("/https/api.example.com/v1/v1/models").unwrap();
        assert_eq!(route.claude_endpoint, "v1/models");
    }

    #[test]
    fn test_parse_dynamic_route_token_counting() {
        let route =
            parse_dynamic_route("/https/api.example.com/v1/v1/messages/count_tokens").unwrap();
        assert_eq!(route.claude_endpoint, "v1/messages/count_tokens");
    }

    #[test]
    fn test_parse_dynamic_route_http_protocol() {
        let route = parse_dynamic_route("/http/localhost/v1/v1/messages").unwrap();
        assert_eq!(route.target_config.target_url, "http://localhost");
    }

    #[test]
    fn test_parse_dynamic_route_invalid_protocol() {
        let result = parse_dynamic_route("/ftp/example.com/v1/v1/messages");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dynamic_route_too_short() {
        let result = parse_dynamic_route("/https/example.com");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dynamic_route_no_claude_endpoint() {
        let result = parse_dynamic_route("/https/example.com/some/random/path");
        assert!(result.is_err());
    }

    // --- build_target_url ---

    #[test]
    fn test_build_target_url_without_model() {
        let config = TargetConfig {
            target_url: "https://api.example.com".to_string(),
            target_path_prefix: "/v1".to_string(),
        };
        let url = build_target_url(&config, "v1/chat/completions", None);
        assert_eq!(url, "https://api.example.com/v1/v1/chat/completions");
    }

    #[test]
    fn test_build_target_url_with_model() {
        let config = TargetConfig {
            target_url: "https://api.example.com".to_string(),
            target_path_prefix: "/v1".to_string(),
        };
        let url = build_target_url(&config, "v1/chat/completions", Some("gpt-4"));
        assert_eq!(url, "https://api.example.com/v1/gpt-4/v1/chat/completions");
    }

    #[test]
    fn test_build_target_url_empty_prefix() {
        let config = TargetConfig {
            target_url: "https://api.example.com".to_string(),
            target_path_prefix: String::new(),
        };
        let url = build_target_url(&config, "v1/models", None);
        assert_eq!(url, "https://api.example.com/v1/models");
    }

    // --- get_handler_type ---

    #[test]
    fn test_get_handler_type_models() {
        assert_eq!(get_handler_type("v1/models").unwrap(), HandlerType::Models);
    }

    #[test]
    fn test_get_handler_type_messages() {
        assert_eq!(
            get_handler_type("v1/messages").unwrap(),
            HandlerType::Messages
        );
    }

    #[test]
    fn test_get_handler_type_token_counting() {
        assert_eq!(
            get_handler_type("v1/messages/count_tokens").unwrap(),
            HandlerType::TokenCounting
        );
    }

    #[test]
    fn test_get_handler_type_unknown() {
        assert!(get_handler_type("v1/unknown").is_err());
    }

    // --- is_host_allowed ---

    #[test]
    fn test_is_host_allowed_exact_match() {
        let hosts = vec!["api.example.com".to_string()];
        assert!(is_host_allowed("api.example.com", &hosts));
    }

    #[test]
    fn test_is_host_allowed_case_insensitive() {
        let hosts = vec!["API.Example.COM".to_string()];
        assert!(is_host_allowed("api.example.com", &hosts));
    }

    #[test]
    fn test_is_host_allowed_wildcard() {
        let hosts = vec!["*.example.com".to_string()];
        assert!(is_host_allowed("api.example.com", &hosts));
        // "example.com" also matches because ends_with("example.com") is true
        assert!(is_host_allowed("example.com", &hosts));
        assert!(!is_host_allowed("evil.com", &hosts));
    }

    #[test]
    fn test_is_host_allowed_not_allowed() {
        let hosts = vec!["api.example.com".to_string()];
        assert!(!is_host_allowed("evil.com", &hosts));
    }

    #[test]
    fn test_is_host_allowed_empty_uses_defaults() {
        let hosts: Vec<String> = vec![];
        assert!(is_host_allowed("localhost", &hosts));
        assert!(is_host_allowed("127.0.0.1", &hosts));
        assert!(!is_host_allowed("evil.com", &hosts));
    }

    // --- extract_auth_headers ---

    #[test]
    fn test_extract_auth_headers_authorization() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-test".parse().unwrap());
        let result = extract_auth_headers(&headers);
        assert_eq!(result.get("Authorization").unwrap(), "Bearer sk-test");
    }

    #[test]
    fn test_extract_auth_headers_x_api_key_converted() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-test".parse().unwrap());
        let result = extract_auth_headers(&headers);
        assert_eq!(result.get("Authorization").unwrap(), "Bearer sk-test");
    }

    #[test]
    fn test_extract_auth_headers_x_api_key_bearer_passthrough() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "Bearer sk-test".parse().unwrap());
        let result = extract_auth_headers(&headers);
        assert_eq!(result.get("Authorization").unwrap(), "Bearer sk-test");
    }

    #[test]
    fn test_extract_auth_headers_authorization_takes_priority() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer from-auth".parse().unwrap());
        headers.insert("x-api-key", "sk-from-key".parse().unwrap());
        let result = extract_auth_headers(&headers);
        assert_eq!(result.get("Authorization").unwrap(), "Bearer from-auth");
    }
}
