use crate::error::ProxyError;
use crate::types::claude::ThinkingConfig;

/// Validate thinking budget.
pub fn validate_thinking_budget(
    thinking: &Option<ThinkingConfig>,
    max_tokens: Option<u32>,
) -> Result<(), ProxyError> {
    if let Some(ThinkingConfig::Enabled { budget_tokens }) = thinking {
        if *budget_tokens < 1024 {
            return Err(ProxyError::Validation(
                "thinking.budget_tokens must be at least 1,024".into(),
            ));
        }
        if *budget_tokens > 100_000 {
            return Err(ProxyError::Validation(
                "thinking.budget_tokens cannot exceed 100,000".into(),
            ));
        }
        if let Some(max) = max_tokens {
            if *budget_tokens > max {
                return Err(ProxyError::Validation(
                    "thinking.budget_tokens cannot exceed max_tokens".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Get effective thinking budget.
pub fn get_effective_thinking_budget(thinking: &Option<ThinkingConfig>) -> Option<u32> {
    match thinking {
        Some(ThinkingConfig::Enabled { budget_tokens }) => Some(*budget_tokens),
        _ => None,
    }
}

/// Check if thinking is enabled.
pub fn is_thinking_enabled(thinking: &Option<ThinkingConfig>) -> bool {
    matches!(thinking, Some(ThinkingConfig::Enabled { .. }))
}

/// Adjust thinking budget based on available tokens.
pub fn adjust_thinking_budget(
    thinking: &Option<ThinkingConfig>,
    available_tokens: u32,
) -> Option<ThinkingConfig> {
    match thinking {
        Some(ThinkingConfig::Enabled { budget_tokens }) => {
            let adjusted = (*budget_tokens).min(available_tokens);
            Some(ThinkingConfig::Enabled {
                budget_tokens: adjusted,
            })
        }
        other => other.clone(),
    }
}

/// Estimate thinking token usage.
pub fn estimate_thinking_tokens(thinking: &Option<ThinkingConfig>) -> u32 {
    match thinking {
        Some(ThinkingConfig::Enabled { budget_tokens }) => *budget_tokens,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled(budget: u32) -> Option<ThinkingConfig> {
        Some(ThinkingConfig::Enabled {
            budget_tokens: budget,
        })
    }

    fn disabled() -> Option<ThinkingConfig> {
        Some(ThinkingConfig::Disabled)
    }

    // --- validate_thinking_budget ---

    #[test]
    fn test_validate_thinking_budget_valid() {
        assert!(validate_thinking_budget(&enabled(2048), Some(10000)).is_ok());
    }

    #[test]
    fn test_validate_thinking_budget_too_low() {
        assert!(validate_thinking_budget(&enabled(512), None).is_err());
    }

    #[test]
    fn test_validate_thinking_budget_too_high() {
        assert!(validate_thinking_budget(&enabled(200_000), None).is_err());
    }

    #[test]
    fn test_validate_thinking_budget_exceeds_max_tokens() {
        assert!(validate_thinking_budget(&enabled(5000), Some(4000)).is_err());
    }

    #[test]
    fn test_validate_thinking_budget_disabled() {
        assert!(validate_thinking_budget(&disabled(), Some(100)).is_ok());
    }

    #[test]
    fn test_validate_thinking_budget_none() {
        assert!(validate_thinking_budget(&None, Some(100)).is_ok());
    }

    // --- get_effective_thinking_budget ---

    #[test]
    fn test_get_effective_budget_enabled() {
        assert_eq!(get_effective_thinking_budget(&enabled(5000)), Some(5000));
    }

    #[test]
    fn test_get_effective_budget_disabled() {
        assert_eq!(get_effective_thinking_budget(&disabled()), None);
    }

    #[test]
    fn test_get_effective_budget_none() {
        assert_eq!(get_effective_thinking_budget(&None), None);
    }

    // --- is_thinking_enabled ---

    #[test]
    fn test_is_thinking_enabled_true() {
        assert!(is_thinking_enabled(&enabled(5000)));
    }

    #[test]
    fn test_is_thinking_enabled_false_disabled() {
        assert!(!is_thinking_enabled(&disabled()));
    }

    #[test]
    fn test_is_thinking_enabled_false_none() {
        assert!(!is_thinking_enabled(&None));
    }

    // --- adjust_thinking_budget ---

    #[test]
    fn test_adjust_thinking_budget_caps() {
        let result = adjust_thinking_budget(&enabled(10000), 5000);
        match result {
            Some(ThinkingConfig::Enabled { budget_tokens }) => assert_eq!(budget_tokens, 5000),
            _ => panic!("Expected Enabled"),
        }
    }

    #[test]
    fn test_adjust_thinking_budget_no_cap_needed() {
        let result = adjust_thinking_budget(&enabled(3000), 5000);
        match result {
            Some(ThinkingConfig::Enabled { budget_tokens }) => assert_eq!(budget_tokens, 3000),
            _ => panic!("Expected Enabled"),
        }
    }

    #[test]
    fn test_adjust_thinking_budget_disabled_passthrough() {
        let result = adjust_thinking_budget(&disabled(), 5000);
        assert!(matches!(result, Some(ThinkingConfig::Disabled)));
    }

    #[test]
    fn test_adjust_thinking_budget_none_passthrough() {
        assert!(adjust_thinking_budget(&None, 5000).is_none());
    }

    // --- estimate_thinking_tokens ---

    #[test]
    fn test_estimate_thinking_tokens_enabled() {
        assert_eq!(estimate_thinking_tokens(&enabled(8000)), 8000);
    }

    #[test]
    fn test_estimate_thinking_tokens_disabled() {
        assert_eq!(estimate_thinking_tokens(&disabled()), 0);
    }

    #[test]
    fn test_estimate_thinking_tokens_none() {
        assert_eq!(estimate_thinking_tokens(&None), 0);
    }
}
