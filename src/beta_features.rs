pub const VALID_BETA_FEATURES: &[&str] = &[
    "message-batches-2024-09-24",
    "prompt-caching-2024-07-31",
    "computer-use-2024-10-22",
    "computer-use-2025-01-24",
    "pdfs-2024-09-25",
    "token-counting-2024-11-01",
    "token-efficient-tools-2025-02-19",
    "output-128k-2025-02-19",
    "files-api-2025-04-14",
    "mcp-client-2025-04-04",
    "mcp-client-2025-11-20",
    "dev-full-thinking-2025-05-14",
    "interleaved-thinking-2025-05-14",
    "code-execution-2025-05-22",
    "extended-cache-ttl-2025-04-11",
    "context-1m-2025-08-07",
    "context-management-2025-06-27",
    "model-context-window-exceeded-2025-08-26",
    "skills-2025-10-02",
];

/// Parse and validate anthropic-beta header (JSON array).
/// Returns Some(features) on success, None on parse failure.
/// Unknown features are silently forwarded.
pub fn validate_beta_features(header_value: &str) -> Option<Vec<String>> {
    let features: Vec<serde_json::Value> = serde_json::from_str(header_value).ok()?;

    let result: Vec<String> = features
        .into_iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();

    if result.is_empty() {
        return None;
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_beta_features_valid_json_array() {
        let input = r#"["prompt-caching-2024-07-31","pdfs-2024-09-25"]"#;
        let result = validate_beta_features(input);
        assert_eq!(
            result,
            Some(vec![
                "prompt-caching-2024-07-31".to_string(),
                "pdfs-2024-09-25".to_string()
            ])
        );
    }

    #[test]
    fn test_validate_beta_features_unknown_features_forwarded() {
        let input = r#"["unknown-feature-2099-01-01"]"#;
        let result = validate_beta_features(input);
        assert_eq!(
            result,
            Some(vec!["unknown-feature-2099-01-01".to_string()])
        );
    }

    #[test]
    fn test_validate_beta_features_empty_array() {
        let result = validate_beta_features("[]");
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_beta_features_invalid_json() {
        let result = validate_beta_features("not-json");
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_beta_features_non_string_values_filtered() {
        let input = r#"["valid-feature", 123, null]"#;
        let result = validate_beta_features(input);
        assert_eq!(result, Some(vec!["valid-feature".to_string()]));
    }

    #[test]
    fn test_valid_beta_features_list_not_empty() {
        assert!(!VALID_BETA_FEATURES.is_empty());
        assert!(VALID_BETA_FEATURES.contains(&"prompt-caching-2024-07-31"));
    }
}
