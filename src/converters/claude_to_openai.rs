use serde_json::Value;

use crate::types::claude::{
    ClaudeContent, ClaudeContentBlock, ClaudeMessagesRequest, ClaudeTokenCountingRequest,
    ClaudeTool, SystemPrompt, ThinkingConfig, ToolChoice,
};
use crate::types::openai::{
    ImageUrlDetail, OpenAIContent, OpenAIContentPart, OpenAIFunctionCall, OpenAIFunctionDef,
    OpenAIMessage, OpenAIRequest, OpenAIThinking, OpenAITokenCountingRequest, OpenAIToolCall,
    OpenAIToolChoice, OpenAIToolChoiceFunction, OpenAIToolChoiceObject, OpenAIToolDef,
};

/// Recursively clean a JSON Schema for compatibility with target APIs.
/// Removes `$schema` and `additionalProperties` keys.
/// For properties of type "string", removes `format` unless it's "date-time" or "enum".
pub fn recursively_clean_schema(schema: &Value) -> Value {
    match schema {
        Value::Array(arr) => Value::Array(arr.iter().map(recursively_clean_schema).collect()),
        Value::Object(obj) => {
            let mut new_obj = serde_json::Map::new();
            for (key, value) in obj {
                if key == "$schema" || key == "additionalProperties" {
                    continue;
                }
                new_obj.insert(key.clone(), recursively_clean_schema(value));
            }
            // For string type properties, remove unsupported format fields
            if new_obj.get("type").and_then(|v| v.as_str()) == Some("string") {
                if let Some(format) = new_obj.get("format").and_then(|v| v.as_str()) {
                    let supported = ["date-time", "enum"];
                    if !supported.contains(&format) {
                        new_obj.remove("format");
                    }
                }
            }
            Value::Object(new_obj)
        }
        other => other.clone(),
    }
}

/// Convert Claude content blocks to OpenAI format.
fn convert_claude_content_to_openai(content: &ClaudeContent) -> OpenAIContent {
    match content {
        ClaudeContent::Text(s) => OpenAIContent::Text(s.clone()),
        ClaudeContent::Blocks(blocks) => {
            let parts: Vec<OpenAIContentPart> = blocks
                .iter()
                .filter_map(|block| match block {
                    ClaudeContentBlock::Text { text, .. } => {
                        Some(OpenAIContentPart::Text { text: text.clone() })
                    }
                    ClaudeContentBlock::Image { source } => {
                        if source.source_type == "base64" {
                            let media_type = source.media_type.as_deref().unwrap_or("image/png");
                            let data = source.data.as_deref().unwrap_or("");
                            Some(OpenAIContentPart::ImageUrl {
                                image_url: ImageUrlDetail {
                                    url: format!("data:{media_type};base64,{data}"),
                                    detail: None,
                                },
                            })
                        } else {
                            None
                        }
                    }
                    _ => None, // Skip document, tool_use, tool_result, etc.
                })
                .collect();

            if parts.is_empty() {
                OpenAIContent::Text(String::new())
            } else {
                OpenAIContent::Parts(parts)
            }
        }
    }
}

/// Convert Claude tools to OpenAI format.
fn convert_claude_tools_to_openai(tools: &Option<Vec<ClaudeTool>>) -> Option<Vec<OpenAIToolDef>> {
    let tools = tools.as_ref()?;
    if tools.is_empty() {
        return None;
    }
    Some(
        tools
            .iter()
            .map(|tool| OpenAIToolDef {
                tool_type: "function".to_string(),
                function: OpenAIFunctionDef {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone().unwrap_or_default()),
                    parameters: recursively_clean_schema(&tool.input_schema),
                },
            })
            .collect(),
    )
}

/// Convert Claude tool_choice to OpenAI format.
fn convert_claude_tool_choice_to_openai(
    tool_choice: &Option<ToolChoice>,
) -> Option<OpenAIToolChoice> {
    let tc = tool_choice.as_ref()?;
    match tc {
        ToolChoice::Auto | ToolChoice::Any => Some(OpenAIToolChoice::String("auto".to_string())),
        ToolChoice::Tool { name } => Some(OpenAIToolChoice::Object(OpenAIToolChoiceObject {
            choice_type: "function".to_string(),
            function: OpenAIToolChoiceFunction { name: name.clone() },
        })),
        ToolChoice::None => Some(OpenAIToolChoice::String("none".to_string())),
    }
}

/// Convert Claude thinking config to OpenAI format.
fn convert_claude_thinking_to_openai(thinking: &Option<ThinkingConfig>) -> Option<OpenAIThinking> {
    let t = thinking.as_ref()?;
    match t {
        ThinkingConfig::Enabled { budget_tokens } => Some(OpenAIThinking {
            enabled: Some(true),
            budget_tokens: Some(*budget_tokens),
        }),
        ThinkingConfig::Disabled => Some(OpenAIThinking {
            enabled: Some(false),
            budget_tokens: None,
        }),
    }
}

/// Extract tool results from a Claude content block array.
fn extract_tool_results(blocks: &[ClaudeContentBlock]) -> Vec<(String, String)> {
    blocks
        .iter()
        .filter_map(|block| {
            if let ClaudeContentBlock::ToolResult {
                tool_use_id,
                content,
            } = block
            {
                let content_str = match content {
                    ClaudeContent::Text(s) => s.clone(),
                    ClaudeContent::Blocks(inner) => {
                        serde_json::to_string(inner).unwrap_or_default()
                    }
                };
                Some((tool_use_id.clone(), content_str))
            } else {
                None
            }
        })
        .collect()
}

/// Extract non-tool-result content blocks.
fn extract_non_tool_content(blocks: &[ClaudeContentBlock]) -> Vec<ClaudeContentBlock> {
    blocks
        .iter()
        .filter(|b| !matches!(b, ClaudeContentBlock::ToolResult { .. }))
        .cloned()
        .collect()
}

/// Process messages common to both messages and token counting requests.
fn convert_messages(
    messages: &[crate::types::claude::ClaudeMessage],
    system: &Option<SystemPrompt>,
) -> Vec<OpenAIMessage> {
    let mut openai_messages: Vec<OpenAIMessage> = Vec::new();

    // System message first
    if let Some(sys) = system {
        let system_content = match sys {
            SystemPrompt::Text(s) => s.clone(),
            SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        openai_messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: OpenAIContent::Text(system_content),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for message in messages {
        if message.role == "user" {
            match &message.content {
                ClaudeContent::Text(s) => {
                    openai_messages.push(OpenAIMessage {
                        role: "user".to_string(),
                        content: OpenAIContent::Text(s.clone()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                ClaudeContent::Blocks(blocks) => {
                    // Extract tool results as separate tool messages
                    let tool_results = extract_tool_results(blocks);
                    for (tool_use_id, content) in &tool_results {
                        openai_messages.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: OpenAIContent::Text(content.clone()),
                            name: None,
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id.clone()),
                        });
                    }

                    // Add other content as user message
                    let other_content = extract_non_tool_content(blocks);
                    if !other_content.is_empty() {
                        let converted =
                            convert_claude_content_to_openai(&ClaudeContent::Blocks(other_content));
                        // Only add if not empty
                        let is_empty = matches!(&converted, OpenAIContent::Text(s) if s.is_empty());
                        if !is_empty {
                            openai_messages.push(OpenAIMessage {
                                role: "user".to_string(),
                                content: converted,
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                    }
                }
            }
        } else if message.role == "assistant" {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();

            if let ClaudeContent::Blocks(blocks) = &message.content {
                for block in blocks {
                    match block {
                        ClaudeContentBlock::Text { text, .. } => {
                            text_parts.push(text.clone());
                        }
                        ClaudeContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(OpenAIToolCall {
                                id: id.clone(),
                                call_type: "function".to_string(),
                                function: OpenAIFunctionCall {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                        _ => {}
                    }
                }
            }

            let content_str = if text_parts.is_empty() {
                String::new()
            } else {
                text_parts.join("\n")
            };

            let mut msg = OpenAIMessage {
                role: "assistant".to_string(),
                content: OpenAIContent::Text(content_str),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            };

            if !tool_calls.is_empty() {
                msg.tool_calls = Some(tool_calls);
            }

            openai_messages.push(msg);
        }
    }

    openai_messages
}

/// Convert a Claude Messages Request to OpenAI format.
pub fn convert_claude_to_openai_request(
    claude_req: &ClaudeMessagesRequest,
    model_name: &str,
) -> OpenAIRequest {
    let openai_messages = convert_messages(&claude_req.messages, &claude_req.system);

    let mut openai_request = OpenAIRequest {
        model: model_name.to_string(),
        messages: openai_messages,
        max_tokens: Some(claude_req.max_tokens),
        temperature: claude_req.temperature,
        top_p: claude_req.top_p,
        stop: claude_req.stop_sequences.clone(),
        stream: claude_req.stream,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    openai_request.tools = convert_claude_tools_to_openai(&claude_req.tools);
    openai_request.tool_choice = convert_claude_tool_choice_to_openai(&claude_req.tool_choice);
    openai_request.thinking = convert_claude_thinking_to_openai(&claude_req.thinking);

    openai_request
}

/// Convert a Claude Token Counting Request to OpenAI format.
pub fn convert_claude_token_counting_to_openai(
    claude_req: &ClaudeTokenCountingRequest,
    model_name: &str,
) -> OpenAITokenCountingRequest {
    let openai_messages = convert_messages(&claude_req.messages, &claude_req.system);

    let mut openai_request = OpenAITokenCountingRequest {
        model: model_name.to_string(),
        messages: openai_messages,
        tools: None,
        thinking: None,
    };

    openai_request.tools = convert_claude_tools_to_openai(&claude_req.tools);
    openai_request.thinking = convert_claude_thinking_to_openai(&claude_req.thinking);

    openai_request
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude::*;
    use crate::types::openai::*;
    use serde_json::json;

    fn simple_request() -> ClaudeMessagesRequest {
        ClaudeMessagesRequest {
            model: "claude-3".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
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

    // --- recursively_clean_schema ---

    #[test]
    fn test_clean_schema_removes_dollar_schema() {
        let schema = json!({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"});
        let cleaned = recursively_clean_schema(&schema);
        assert!(cleaned.get("$schema").is_none());
        assert_eq!(cleaned.get("type").unwrap(), "object");
    }

    #[test]
    fn test_clean_schema_removes_additional_properties() {
        let schema = json!({"type": "object", "additionalProperties": false});
        let cleaned = recursively_clean_schema(&schema);
        assert!(cleaned.get("additionalProperties").is_none());
    }

    #[test]
    fn test_clean_schema_removes_unsupported_format() {
        let schema = json!({"type": "string", "format": "email"});
        let cleaned = recursively_clean_schema(&schema);
        assert!(cleaned.get("format").is_none());
    }

    #[test]
    fn test_clean_schema_keeps_supported_format() {
        let schema = json!({"type": "string", "format": "date-time"});
        let cleaned = recursively_clean_schema(&schema);
        assert_eq!(cleaned.get("format").unwrap(), "date-time");
    }

    #[test]
    fn test_clean_schema_recursive() {
        let schema = json!({
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "name": {"type": "string"}
            },
            "$schema": "test"
        });
        let cleaned = recursively_clean_schema(&schema);
        assert!(cleaned.get("$schema").is_none());
        let email = cleaned.get("properties").unwrap().get("email").unwrap();
        assert!(email.get("format").is_none());
    }

    // --- convert_claude_to_openai_request ---

    #[test]
    fn test_convert_basic_request() {
        let req = simple_request();
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        assert_eq!(openai.model, "gpt-4");
        assert_eq!(openai.messages.len(), 1);
        assert_eq!(openai.messages[0].role, "user");
        assert_eq!(openai.max_tokens, Some(1024));
    }

    #[test]
    fn test_convert_with_system_prompt_text() {
        let mut req = simple_request();
        req.system = Some(SystemPrompt::Text("You are helpful.".to_string()));
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        assert_eq!(openai.messages.len(), 2);
        assert_eq!(openai.messages[0].role, "system");
        match &openai.messages[0].content {
            OpenAIContent::Text(t) => assert_eq!(t, "You are helpful."),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_convert_with_system_prompt_blocks() {
        let mut req = simple_request();
        req.system = Some(SystemPrompt::Blocks(vec![
            ClaudeTextBlock {
                block_type: "text".to_string(),
                text: "Line one.".to_string(),
                cache_control: None,
            },
            ClaudeTextBlock {
                block_type: "text".to_string(),
                text: "Line two.".to_string(),
                cache_control: None,
            },
        ]));
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        assert_eq!(openai.messages[0].role, "system");
        match &openai.messages[0].content {
            OpenAIContent::Text(t) => assert_eq!(t, "Line one.\nLine two."),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_convert_with_tools() {
        let mut req = simple_request();
        req.tools = Some(vec![ClaudeTool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: json!({"type": "object", "properties": {"loc": {"type": "string"}}}),
        }]);
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        let tools = openai.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[0].tool_type, "function");
    }

    #[test]
    fn test_convert_tool_choice_auto() {
        let mut req = simple_request();
        req.tool_choice = Some(ToolChoice::Auto);
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        match openai.tool_choice.unwrap() {
            OpenAIToolChoice::String(s) => assert_eq!(s, "auto"),
            _ => panic!("Expected string tool choice"),
        }
    }

    #[test]
    fn test_convert_tool_choice_none() {
        let mut req = simple_request();
        req.tool_choice = Some(ToolChoice::None);
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        match openai.tool_choice.unwrap() {
            OpenAIToolChoice::String(s) => assert_eq!(s, "none"),
            _ => panic!("Expected string tool choice"),
        }
    }

    #[test]
    fn test_convert_tool_choice_specific() {
        let mut req = simple_request();
        req.tool_choice = Some(ToolChoice::Tool {
            name: "my_tool".to_string(),
        });
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        match openai.tool_choice.unwrap() {
            OpenAIToolChoice::Object(obj) => {
                assert_eq!(obj.choice_type, "function");
                assert_eq!(obj.function.name, "my_tool");
            }
            _ => panic!("Expected object tool choice"),
        }
    }

    #[test]
    fn test_convert_thinking_enabled() {
        let mut req = simple_request();
        req.thinking = Some(ThinkingConfig::Enabled {
            budget_tokens: 5000,
        });
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        let thinking = openai.thinking.unwrap();
        assert_eq!(thinking.enabled, Some(true));
        assert_eq!(thinking.budget_tokens, Some(5000));
    }

    #[test]
    fn test_convert_thinking_disabled() {
        let mut req = simple_request();
        req.thinking = Some(ThinkingConfig::Disabled);
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        let thinking = openai.thinking.unwrap();
        assert_eq!(thinking.enabled, Some(false));
        assert_eq!(thinking.budget_tokens, None);
    }

    #[test]
    fn test_convert_assistant_message_with_tool_calls() {
        let mut req = simple_request();
        req.messages = vec![
            ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("What's the weather?".to_string()),
            },
            ClaudeMessage {
                role: "assistant".to_string(),
                content: ClaudeContent::Blocks(vec![
                    ClaudeContentBlock::Text {
                        text: "Let me check.".to_string(),
                        citations: None,
                        cache_control: None,
                    },
                    ClaudeContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        input: json!({"location": "NYC"}),
                    },
                ]),
            },
        ];
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        let assistant_msg = &openai.messages[1];
        assert_eq!(assistant_msg.role, "assistant");
        let tool_calls = assistant_msg.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_convert_user_message_with_tool_result() {
        let mut req = simple_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![
                ClaudeContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: ClaudeContent::Text("Sunny, 72F".to_string()),
                },
                ClaudeContentBlock::Text {
                    text: "Thanks!".to_string(),
                    citations: None,
                    cache_control: None,
                },
            ]),
        }];
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        // Should have tool message + user message
        assert!(openai.messages.len() >= 2);
        assert_eq!(openai.messages[0].role, "tool");
        assert_eq!(openai.messages[0].tool_call_id, Some("call_1".to_string()));
    }

    #[test]
    fn test_convert_image_base64() {
        let mut req = simple_request();
        req.messages = vec![ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(vec![ClaudeContentBlock::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: Some("image/png".to_string()),
                    data: Some("abc123".to_string()),
                    url: None,
                },
            }]),
        }];
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        match &openai.messages[0].content {
            OpenAIContent::Parts(parts) => match &parts[0] {
                OpenAIContentPart::ImageUrl { image_url } => {
                    assert_eq!(image_url.url, "data:image/png;base64,abc123");
                }
                _ => panic!("Expected ImageUrl part"),
            },
            _ => panic!("Expected Parts content"),
        }
    }

    // --- convert_claude_token_counting_to_openai ---

    #[test]
    fn test_convert_token_counting_request() {
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
        let openai = convert_claude_token_counting_to_openai(&req, "gpt-4");
        assert_eq!(openai.model, "gpt-4");
        assert_eq!(openai.messages.len(), 1);
    }

    #[test]
    fn test_convert_with_stop_sequences() {
        let mut req = simple_request();
        req.stop_sequences = Some(vec!["STOP".to_string()]);
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        assert_eq!(openai.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_convert_with_sampling_params() {
        let mut req = simple_request();
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.stream = Some(true);
        let openai = convert_claude_to_openai_request(&req, "gpt-4");
        assert_eq!(openai.temperature, Some(0.7));
        assert_eq!(openai.top_p, Some(0.9));
        assert_eq!(openai.stream, Some(true));
    }
}
