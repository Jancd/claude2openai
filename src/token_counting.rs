use once_cell::sync::OnceCell;
use tiktoken_rs::CoreBPE;

use crate::types::claude::{
    ClaudeContent, ClaudeContentBlock, ClaudeTokenCountingRequest, SystemPrompt,
};

static TOKENIZER: OnceCell<CoreBPE> = OnceCell::new();

/// Initialize or get the cached tiktoken tokenizer.
pub fn get_tokenizer(model_name: &str) -> Option<&'static CoreBPE> {
    TOKENIZER
        .get_or_try_init(|| {
            tiktoken_rs::get_bpe_from_model(model_name).or_else(|_| tiktoken_rs::cl100k_base())
        })
        .ok()
}

/// Estimate token count using character-based approximation (default 4 chars/token).
pub fn estimate_token_count(text: &str, chars_per_token: usize) -> usize {
    if text.is_empty() {
        return 0;
    }
    let estimated = (text.len() + chars_per_token - 1) / chars_per_token;
    estimated + 5 // overhead
}

/// Count tokens using tiktoken BPE encoding.
pub fn count_tokens_tiktoken(text: &str, tokenizer: &CoreBPE) -> usize {
    if text.is_empty() {
        return 0;
    }
    tokenizer.encode_with_special_tokens(text).len()
}

/// Count tokens in a string, using tiktoken if available or estimation.
fn count_string_tokens(text: &str, tokenizer: Option<&CoreBPE>) -> usize {
    match tokenizer {
        Some(tok) => count_tokens_tiktoken(text, tok),
        None => estimate_token_count(text, 4),
    }
}

/// Extract text from Claude content for token counting.
fn extract_content_text(content: &ClaudeContent) -> String {
    match content {
        ClaudeContent::Text(s) => s.clone(),
        ClaudeContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ClaudeContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Count tokens in a Claude request.
pub fn count_claude_request_tokens(
    request: &ClaudeTokenCountingRequest,
    tokenizer: Option<&CoreBPE>,
) -> usize {
    let mut total: usize = 0;

    // Count model name
    total += count_string_tokens(&request.model, tokenizer) + 1;

    // Count system prompt
    if let Some(ref system) = request.system {
        match system {
            SystemPrompt::Text(s) => {
                total += count_string_tokens(s, tokenizer);
            }
            SystemPrompt::Blocks(blocks) => {
                for block in blocks {
                    total += count_string_tokens(&block.text, tokenizer);
                }
            }
        }
    }

    // Count messages
    for msg in &request.messages {
        // role tokens
        let role_text = format!("role: {}", msg.role);
        total += count_string_tokens(&role_text, tokenizer);

        // content tokens
        let content_text = extract_content_text(&msg.content);
        total += count_string_tokens(&content_text, tokenizer);

        // overhead per message
        total += 2;
    }

    // Message separators overhead
    total += 3;

    // Tools
    if let Some(ref tools) = request.tools {
        for tool in tools {
            let mut tool_content = format!("tool: {}", tool.name);
            if let Some(ref desc) = tool.description {
                tool_content.push_str(&format!("\ndescription: {desc}"));
            }
            tool_content.push_str(&format!(
                "\nparameters: {}",
                serde_json::to_string(&tool.input_schema).unwrap_or_default()
            ));
            total += count_string_tokens(&tool_content, tokenizer) + 3;
        }
        total += 5; // tools array overhead
    }

    // Tool choice
    if let Some(ref tc) = request.tool_choice {
        let tc_text = match tc {
            crate::types::claude::ToolChoice::Tool { name } => {
                format!("tool_choice: {name}")
            }
            crate::types::claude::ToolChoice::Auto => "tool_choice: auto".to_string(),
            crate::types::claude::ToolChoice::Any => "tool_choice: any".to_string(),
            crate::types::claude::ToolChoice::None => "tool_choice: none".to_string(),
        };
        total += count_string_tokens(&tc_text, tokenizer) + 2;
    }

    // Thinking
    if let Some(ref thinking) = request.thinking {
        match thinking {
            crate::types::claude::ThinkingConfig::Enabled { budget_tokens } => {
                total += count_string_tokens("thinking: enabled", tokenizer) + 2;
                total += count_string_tokens(&format!("budget: {budget_tokens}"), tokenizer);
            }
            crate::types::claude::ThinkingConfig::Disabled => {
                total += count_string_tokens("thinking: disabled", tokenizer) + 2;
            }
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude::*;

    // --- estimate_token_count ---

    #[test]
    fn test_estimate_token_count_empty() {
        assert_eq!(estimate_token_count("", 4), 0);
    }

    #[test]
    fn test_estimate_token_count_short() {
        // "hello" = 5 chars, (5+3)/4 = 2 + 5 overhead = 7
        assert_eq!(estimate_token_count("hello", 4), 7);
    }

    #[test]
    fn test_estimate_token_count_exact_multiple() {
        // "abcd" = 4 chars, (4+3)/4 = 1 + 5 = 6
        assert_eq!(estimate_token_count("abcd", 4), 6);
    }

    // --- count_tokens_tiktoken ---

    #[test]
    fn test_count_tokens_tiktoken_empty() {
        let tokenizer = tiktoken_rs::cl100k_base().unwrap();
        assert_eq!(count_tokens_tiktoken("", &tokenizer), 0);
    }

    #[test]
    fn test_count_tokens_tiktoken_hello() {
        let tokenizer = tiktoken_rs::cl100k_base().unwrap();
        let count = count_tokens_tiktoken("hello", &tokenizer);
        assert!(count > 0);
    }

    // --- count_claude_request_tokens ---

    #[test]
    fn test_count_request_tokens_minimal() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        let count = count_claude_request_tokens(&req, None);
        assert!(count > 0);
    }

    #[test]
    fn test_count_request_tokens_with_system() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: Some(SystemPrompt::Text("You are helpful.".to_string())),
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        let count_with_system = count_claude_request_tokens(&req, None);

        let req_no_system = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        let count_without_system = count_claude_request_tokens(&req_no_system, None);

        assert!(count_with_system > count_without_system);
    }

    #[test]
    fn test_count_request_tokens_with_tools() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: None,
            tools: Some(vec![ClaudeTool {
                name: "get_weather".to_string(),
                description: Some("Get weather for a location".to_string()),
                input_schema: serde_json::json!({"type": "object", "properties": {"location": {"type": "string"}}}),
            }]),
            tool_choice: None,
            thinking: None,
        };
        let count = count_claude_request_tokens(&req, None);
        assert!(count > 0);
    }

    #[test]
    fn test_count_request_tokens_with_thinking() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: Some(ThinkingConfig::Enabled {
                budget_tokens: 5000,
            }),
        };
        let count = count_claude_request_tokens(&req, None);
        assert!(count > 0);
    }

    #[test]
    fn test_count_request_tokens_with_tiktoken() {
        let tokenizer = tiktoken_rs::cl100k_base().unwrap();
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello world".to_string()),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        let count = count_claude_request_tokens(&req, Some(&tokenizer));
        assert!(count > 0);
    }

    #[test]
    fn test_count_request_tokens_block_content() {
        let req = ClaudeTokenCountingRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock::Text {
                    text: "hello world".to_string(),
                    citations: None,
                    cache_control: None,
                }]),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        let count = count_claude_request_tokens(&req, None);
        assert!(count > 0);
    }
}
