use crate::error::ProxyError;
use crate::types::claude::{
    ClaudeContent, ClaudeContentBlock, ClaudeMessagesRequest, ClaudeTokenCountingRequest,
    ThinkingConfig,
};

const DEFAULT_IMAGE_DATA_MAX_SIZE: usize = 10 * 1024 * 1024;

/// Validate a Claude Messages Request.
pub fn validate_claude_messages_request(
    request: &ClaudeMessagesRequest,
    model_id: Option<&str>,
    max_image_data_size: usize,
) -> Result<(), ProxyError> {
    // Validate messages
    if request.messages.is_empty() {
        return Err(ProxyError::Validation(
            "messages array must not be empty".into(),
        ));
    }
    if request.messages.len() > 100_000 {
        return Err(ProxyError::Validation(
            "messages array cannot exceed 100,000 messages per request".into(),
        ));
    }

    for (i, msg) in request.messages.iter().enumerate() {
        validate_claude_message(msg, &format!("messages[{i}]"), max_image_data_size)?;
    }

    // Validate model
    if model_id.is_none() && request.model.is_empty() {
        return Err(ProxyError::Validation(
            "Either model must be specified in URL or in request body".into(),
        ));
    }

    // Validate max_tokens
    if request.max_tokens < 1 {
        return Err(ProxyError::Validation(
            "max_tokens must be at least 1".into(),
        ));
    }
    if request.max_tokens > 100_000 {
        return Err(ProxyError::Validation(
            "max_tokens cannot exceed 100,000".into(),
        ));
    }

    // Validate temperature
    if let Some(temp) = request.temperature {
        if !(0.0..=1.0).contains(&temp) {
            return Err(ProxyError::Validation(
                "temperature must be between 0 and 1".into(),
            ));
        }
    }

    // Validate top_p
    if let Some(top_p) = request.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ProxyError::Validation(
                "top_p must be between 0 and 1".into(),
            ));
        }
    }

    // Validate top_k
    if let Some(top_k) = request.top_k {
        if !(1..=1000).contains(&top_k) {
            return Err(ProxyError::Validation(
                "top_k must be between 1 and 1000".into(),
            ));
        }
    }

    // Validate thinking
    if let Some(ref thinking) = request.thinking {
        validate_thinking_config(thinking)?;
    }

    // Validate stop_sequences
    if let Some(ref seqs) = request.stop_sequences {
        for (i, seq) in seqs.iter().enumerate() {
            if seq.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "stop_sequences[{i}] must be a non-empty string"
                )));
            }
        }
    }

    Ok(())
}

/// Validate a single Claude message.
fn validate_claude_message(
    message: &crate::types::claude::ClaudeMessage,
    context: &str,
    max_image_data_size: usize,
) -> Result<(), ProxyError> {
    // Validate role
    if message.role != "user" && message.role != "assistant" {
        return Err(ProxyError::Validation(format!(
            "{context}.role must be one of: user, assistant"
        )));
    }

    validate_claude_content(
        &message.content,
        &format!("{context}.content"),
        max_image_data_size,
    )
}

/// Validate Claude content (string or array).
fn validate_claude_content(
    content: &ClaudeContent,
    context: &str,
    max_image_data_size: usize,
) -> Result<(), ProxyError> {
    match content {
        ClaudeContent::Text(s) => {
            if s.trim().is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context} string must not be empty"
                )));
            }
        }
        ClaudeContent::Blocks(blocks) => {
            if blocks.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context} array must not be empty"
                )));
            }
            for (i, block) in blocks.iter().enumerate() {
                validate_claude_content_block(
                    block,
                    &format!("{context}[{i}]"),
                    max_image_data_size,
                )?;
            }
        }
    }
    Ok(())
}

/// Validate a single Claude content block.
fn validate_claude_content_block(
    block: &ClaudeContentBlock,
    context: &str,
    max_image_data_size: usize,
) -> Result<(), ProxyError> {
    match block {
        ClaudeContentBlock::Text { text, .. } => {
            if text.is_empty() {
                // text blocks require a text field, but empty string is technically valid per TS
            }
        }
        ClaudeContentBlock::Image { source } => {
            if source.source_type != "base64" && source.source_type != "url" {
                return Err(ProxyError::Validation(format!(
                    "{context}.source.type must be 'base64' or 'url'"
                )));
            }
            if source.source_type == "base64" {
                if source.media_type.is_none() {
                    return Err(ProxyError::Validation(format!(
                        "{context}.source.media_type is required for base64 images"
                    )));
                }
                match &source.data {
                    Some(data) => {
                        if data.len() > max_image_data_size {
                            return Err(ProxyError::Validation(format!(
                                "{context}.source.data exceeds maximum size of {max_image_data_size} bytes"
                            )));
                        }
                    }
                    None => {
                        return Err(ProxyError::Validation(format!(
                            "{context}.source.data is required for base64 images"
                        )));
                    }
                }
            } else if source.source_type == "url" && source.url.is_none() {
                return Err(ProxyError::Validation(format!(
                    "{context}.source.url is required for URL images"
                )));
            }
        }
        ClaudeContentBlock::Document { source, .. } => {
            if source.source_type != "base64" && source.source_type != "text" {
                return Err(ProxyError::Validation(format!(
                    "{context}.source.type must be 'base64' or 'text'"
                )));
            }
        }
        ClaudeContentBlock::ToolUse { id, name, .. } => {
            if id.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context}.id is required for tool_use blocks"
                )));
            }
            if name.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context}.name is required for tool_use blocks"
                )));
            }
        }
        ClaudeContentBlock::ToolResult { tool_use_id, .. } => {
            if tool_use_id.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context}.tool_use_id is required for tool_result blocks"
                )));
            }
        }
        ClaudeContentBlock::WebSearchResult {
            search_query,
            search_results,
        } => {
            if search_query.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context}.search_query is required for web_search_result blocks"
                )));
            }
            if search_results.is_empty() {
                return Err(ProxyError::Validation(format!(
                    "{context}.search_results is required and must be an array"
                )));
            }
        }
        ClaudeContentBlock::Thinking { thinking, .. } => {
            if thinking.is_empty() {
                // thinking text can be empty-ish; TS checks for undefined but not emptiness explicitly
            }
        }
    }
    Ok(())
}

/// Validate thinking configuration.
fn validate_thinking_config(thinking: &ThinkingConfig) -> Result<(), ProxyError> {
    if let ThinkingConfig::Enabled { budget_tokens } = thinking {
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
    }
    Ok(())
}

/// Validate Claude token counting request.
pub fn validate_claude_token_counting_request(
    request: &ClaudeTokenCountingRequest,
    max_image_data_size: usize,
) -> Result<(), ProxyError> {
    if request.model.is_empty() {
        return Err(ProxyError::Validation("model field is required".into()));
    }

    if request.messages.is_empty() {
        return Err(ProxyError::Validation(
            "messages array must not be empty".into(),
        ));
    }

    if request.messages.len() > 100_000 {
        return Err(ProxyError::Validation(
            "messages array cannot exceed 100,000 messages per request".into(),
        ));
    }

    for (i, msg) in request.messages.iter().enumerate() {
        validate_claude_message(msg, &format!("messages[{i}]"), max_image_data_size)?;
    }

    if let Some(ref thinking) = request.thinking {
        validate_thinking_config(thinking)?;
    }

    Ok(())
}

/// Validate models request query parameters.
pub fn validate_models_request_params(limit: Option<u32>) -> Result<(), ProxyError> {
    if let Some(l) = limit {
        if !(1..=1000).contains(&l) {
            return Err(ProxyError::Validation(
                "limit must be between 1 and 1000".into(),
            ));
        }
    }
    Ok(())
}

/// Validate that auth headers are present.
pub fn validate_auth_headers(
    headers: &std::collections::HashMap<String, String>,
) -> Result<(), ProxyError> {
    if !headers.contains_key("Authorization") && !headers.contains_key("x-api-key") {
        return Err(ProxyError::Validation(
            "Either Authorization header or x-api-key header is required".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude::*;
    use std::collections::HashMap;

    fn simple_user_message(text: &str) -> ClaudeMessage {
        ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Text(text.to_string()),
        }
    }

    fn minimal_request() -> ClaudeMessagesRequest {
        ClaudeMessagesRequest {
            model: "claude-3".to_string(),
            messages: vec![simple_user_message("hello")],
            system: None,
            max_tokens: 1024,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            service_tier: None,
            metadata: None,
        }
    }

    // --- validate_claude_messages_request ---

    #[test]
    fn test_validate_valid_request() {
        assert!(validate_claude_messages_request(&minimal_request(), Some("model"), 10_000_000).is_ok());
    }

    #[test]
    fn test_validate_empty_messages() {
        let mut req = minimal_request();
        req.messages = vec![];
        assert!(validate_claude_messages_request(&req, Some("model"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_no_model() {
        let mut req = minimal_request();
        req.model = String::new();
        assert!(validate_claude_messages_request(&req, None, 10_000_000).is_err());
    }

    #[test]
    fn test_validate_model_in_url_overrides_empty_body() {
        let mut req = minimal_request();
        req.model = String::new();
        assert!(validate_claude_messages_request(&req, Some("gpt-4"), 10_000_000).is_ok());
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let mut req = minimal_request();
        req.max_tokens = 0;
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_max_tokens_too_high() {
        let mut req = minimal_request();
        req.max_tokens = 200_000;
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_temperature_out_of_range() {
        let mut req = minimal_request();
        req.temperature = Some(1.5);
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_temperature_valid() {
        let mut req = minimal_request();
        req.temperature = Some(0.7);
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_ok());
    }

    #[test]
    fn test_validate_top_p_out_of_range() {
        let mut req = minimal_request();
        req.top_p = Some(-0.1);
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_top_k_out_of_range() {
        let mut req = minimal_request();
        req.top_k = Some(0);
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_invalid_role() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "system".to_string(),
            content: ClaudeContent::Text("hello".to_string()),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_empty_text_content() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Text("   ".to_string()),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_empty_blocks_content() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![]),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_image_base64_missing_media_type() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![ClaudeContentBlock::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: None,
                    data: Some("abc".to_string()),
                    url: None,
                },
            }]),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_image_base64_missing_data() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![ClaudeContentBlock::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: Some("image/png".to_string()),
                    data: None,
                    url: None,
                },
            }]),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_image_url_missing_url() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![ClaudeContentBlock::Image {
                source: ImageSource {
                    source_type: "url".to_string(),
                    media_type: None,
                    data: None,
                    url: None,
                },
            }]),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_tool_use_missing_id() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![ClaudeContentBlock::ToolUse {
                id: String::new(),
                name: "test".to_string(),
                input: serde_json::json!({}),
            }]),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_tool_result_missing_tool_use_id() {
        let mut req = minimal_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![ClaudeContentBlock::ToolResult {
                tool_use_id: String::new(),
                content: ClaudeContent::Text("result".to_string()),
            }]),
        }];
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_thinking_budget_too_low() {
        let mut req = minimal_request();
        req.thinking = Some(ThinkingConfig::Enabled {
            budget_tokens: 100,
        });
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_thinking_budget_too_high() {
        let mut req = minimal_request();
        req.thinking = Some(ThinkingConfig::Enabled {
            budget_tokens: 200_000,
        });
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    #[test]
    fn test_validate_empty_stop_sequence() {
        let mut req = minimal_request();
        req.stop_sequences = Some(vec!["".to_string()]);
        assert!(validate_claude_messages_request(&req, Some("m"), 10_000_000).is_err());
    }

    // --- validate_claude_token_counting_request ---

    #[test]
    fn test_validate_token_counting_valid() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![simple_user_message("hello")],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        assert!(validate_claude_token_counting_request(&req, 10_000_000).is_ok());
    }

    #[test]
    fn test_validate_token_counting_empty_model() {
        let req = ClaudeTokenCountingRequest {
            model: String::new(),
            messages: vec![simple_user_message("hello")],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        assert!(validate_claude_token_counting_request(&req, 10_000_000).is_err());
    }

    #[test]
    fn test_validate_token_counting_empty_messages() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        assert!(validate_claude_token_counting_request(&req, 10_000_000).is_err());
    }

    // --- validate_models_request_params ---

    #[test]
    fn test_validate_models_params_valid() {
        assert!(validate_models_request_params(Some(100)).is_ok());
    }

    #[test]
    fn test_validate_models_params_none() {
        assert!(validate_models_request_params(None).is_ok());
    }

    #[test]
    fn test_validate_models_params_zero() {
        assert!(validate_models_request_params(Some(0)).is_err());
    }

    #[test]
    fn test_validate_models_params_too_high() {
        assert!(validate_models_request_params(Some(1001)).is_err());
    }

    // --- validate_auth_headers ---

    #[test]
    fn test_validate_auth_headers_with_authorization() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer sk-test".to_string());
        assert!(validate_auth_headers(&headers).is_ok());
    }

    #[test]
    fn test_validate_auth_headers_with_api_key() {
        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), "sk-test".to_string());
        assert!(validate_auth_headers(&headers).is_ok());
    }

    #[test]
    fn test_validate_auth_headers_missing() {
        let headers = HashMap::new();
        assert!(validate_auth_headers(&headers).is_err());
    }
}
