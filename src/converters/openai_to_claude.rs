use chrono::{DateTime, TimeZone, Utc};

use crate::types::claude::{
    ClaudeContentBlock, ClaudeMessagesResponse, ClaudeModel, ClaudeModelsResponse, Usage,
};
use crate::types::openai::{
    OpenAIContent, OpenAIContentPart, OpenAIModelsResponse, OpenAIResponse,
};

/// Convert OpenAI finish_reason to Claude stop_reason.
fn convert_finish_reason_to_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    let reason = finish_reason?;
    let mapped = match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "stop_sequence" => "end_turn",
        "content_filter" => "content_filter",
        _ => "end_turn",
    };
    Some(mapped.to_string())
}

/// Convert an OpenAI Response to a Claude Messages Response.
pub fn convert_openai_to_claude_response(
    openai_resp: &OpenAIResponse,
    model: &str,
    request_id: &str,
) -> ClaudeMessagesResponse {
    // Handle empty choices
    if openai_resp.choices.is_empty() {
        return ClaudeMessagesResponse {
            id: if openai_resp.id.is_empty() {
                request_id.to_string()
            } else {
                openai_resp.id.clone()
            },
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: model.to_string(),
            content: vec![],
            stop_reason: None,
            usage: Usage {
                input_tokens: openai_resp.usage.as_ref().map_or(0, |u| u.prompt_tokens),
                output_tokens: openai_resp
                    .usage
                    .as_ref()
                    .map_or(0, |u| u.completion_tokens),
                cache_creation_input_tokens: openai_resp
                    .usage
                    .as_ref()
                    .and_then(|u| u.prompt_cache_miss_tokens),
                cache_read_input_tokens: openai_resp
                    .usage
                    .as_ref()
                    .and_then(|u| u.prompt_cache_hit_tokens),
            },
        };
    }

    let choice = &openai_resp.choices[0];
    let message = &choice.message;
    let mut content_blocks: Vec<ClaudeContentBlock> = Vec::new();

    // Handle text content
    match &message.content {
        OpenAIContent::Text(text) => {
            if !text.is_empty() {
                content_blocks.push(ClaudeContentBlock::Text {
                    text: text.clone(),
                    citations: None,
                    cache_control: None,
                });
            }
        }
        OpenAIContent::Parts(parts) => {
            let text_content: String = parts
                .iter()
                .filter_map(|part| {
                    if let OpenAIContentPart::Text { text } = part {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            if !text_content.is_empty() {
                content_blocks.push(ClaudeContentBlock::Text {
                    text: text_content,
                    citations: None,
                    cache_control: None,
                });
            }
        }
    }

    // If content was present (even empty string from OpenAI), ensure we have a text block
    let has_content = match &message.content {
        OpenAIContent::Text(s) => !s.is_empty(),
        OpenAIContent::Parts(p) => !p.is_empty(),
    };
    if has_content && content_blocks.is_empty() {
        content_blocks.push(ClaudeContentBlock::Text {
            text: String::new(),
            citations: None,
            cache_control: None,
        });
    }

    // Handle tool calls
    if let Some(tool_calls) = &message.tool_calls {
        for call in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&call.function.arguments)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            content_blocks.push(ClaudeContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.function.name.clone(),
                input,
            });
        }
    }

    let stop_reason = convert_finish_reason_to_stop_reason(choice.finish_reason.as_deref());

    ClaudeMessagesResponse {
        id: if openai_resp.id.is_empty() {
            request_id.to_string()
        } else {
            openai_resp.id.clone()
        },
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: model.to_string(),
        content: content_blocks,
        stop_reason,
        usage: Usage {
            input_tokens: openai_resp.usage.as_ref().map_or(0, |u| u.prompt_tokens),
            output_tokens: openai_resp
                .usage
                .as_ref()
                .map_or(0, |u| u.completion_tokens),
            cache_creation_input_tokens: openai_resp
                .usage
                .as_ref()
                .and_then(|u| u.prompt_cache_miss_tokens),
            cache_read_input_tokens: openai_resp
                .usage
                .as_ref()
                .and_then(|u| u.prompt_cache_hit_tokens),
        },
    }
}

/// Convert Unix timestamp to RFC 3339 string.
fn unix_to_rfc3339(timestamp: u64) -> String {
    let dt: DateTime<Utc> = Utc
        .timestamp_opt(timestamp as i64, 0)
        .single()
        .unwrap_or_else(Utc::now);
    dt.to_rfc3339()
}

/// Convert OpenAI models response to Claude format.
pub fn convert_openai_models_to_claude(openai_resp: &OpenAIModelsResponse) -> ClaudeModelsResponse {
    let models: Vec<ClaudeModel> = openai_resp
        .data
        .iter()
        .map(|m| ClaudeModel {
            id: m.id.clone(),
            model_type: "model".to_string(),
            created_at: unix_to_rfc3339(m.created),
            display_name: m.id.clone(),
        })
        .collect();

    let first_id = models.first().map(|m| m.id.clone());
    let last_id = models.last().map(|m| m.id.clone());

    ClaudeModelsResponse {
        data: models,
        first_id,
        has_more: false,
        last_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::*;

    fn simple_openai_response() -> OpenAIResponse {
        OpenAIResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1700000000,
            model: "gpt-4".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: OpenAIContent::Text("Hello!".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
            }),
            system_fingerprint: None,
        }
    }

    // --- convert_openai_to_claude_response ---

    #[test]
    fn test_convert_basic_response() {
        let resp = simple_openai_response();
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert_eq!(claude.id, "chatcmpl-123");
        assert_eq!(claude.response_type, "message");
        assert_eq!(claude.role, "assistant");
        assert_eq!(claude.model, "claude-3");
        assert_eq!(claude.stop_reason, Some("end_turn".to_string()));
        assert_eq!(claude.usage.input_tokens, 10);
        assert_eq!(claude.usage.output_tokens, 5);
        assert_eq!(claude.content.len(), 1);
        match &claude.content[0] {
            ClaudeContentBlock::Text { text, .. } => assert_eq!(text, "Hello!"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_convert_empty_choices() {
        let mut resp = simple_openai_response();
        resp.choices = vec![];
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert!(claude.content.is_empty());
        assert!(claude.stop_reason.is_none());
    }

    #[test]
    fn test_convert_empty_id_uses_request_id() {
        let mut resp = simple_openai_response();
        resp.id = String::new();
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_fallback");
        assert_eq!(claude.id, "req_fallback");
    }

    #[test]
    fn test_convert_finish_reason_length() {
        let mut resp = simple_openai_response();
        resp.choices[0].finish_reason = Some("length".to_string());
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert_eq!(claude.stop_reason, Some("max_tokens".to_string()));
    }

    #[test]
    fn test_convert_finish_reason_tool_calls() {
        let mut resp = simple_openai_response();
        resp.choices[0].finish_reason = Some("tool_calls".to_string());
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert_eq!(claude.stop_reason, Some("tool_use".to_string()));
    }

    #[test]
    fn test_convert_finish_reason_content_filter() {
        let mut resp = simple_openai_response();
        resp.choices[0].finish_reason = Some("content_filter".to_string());
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert_eq!(claude.stop_reason, Some("content_filter".to_string()));
    }

    #[test]
    fn test_convert_with_tool_calls() {
        let mut resp = simple_openai_response();
        resp.choices[0].message.tool_calls = Some(vec![OpenAIToolCall {
            id: "call_abc".to_string(),
            call_type: "function".to_string(),
            function: OpenAIFunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"NYC"}"#.to_string(),
            },
        }]);
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert!(claude.content.len() >= 2);
        match &claude.content[1] {
            ClaudeContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "get_weather");
                assert_eq!(input.get("location").unwrap(), "NYC");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_convert_with_cache_tokens() {
        let mut resp = simple_openai_response();
        resp.usage = Some(OpenAIUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_cache_hit_tokens: Some(80),
            prompt_cache_miss_tokens: Some(20),
        });
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert_eq!(claude.usage.cache_read_input_tokens, Some(80));
        assert_eq!(claude.usage.cache_creation_input_tokens, Some(20));
    }

    #[test]
    fn test_convert_no_usage() {
        let mut resp = simple_openai_response();
        resp.usage = None;
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        assert_eq!(claude.usage.input_tokens, 0);
        assert_eq!(claude.usage.output_tokens, 0);
    }

    #[test]
    fn test_convert_parts_content() {
        let mut resp = simple_openai_response();
        resp.choices[0].message.content = OpenAIContent::Parts(vec![
            OpenAIContentPart::Text {
                text: "Part 1 ".to_string(),
            },
            OpenAIContentPart::Text {
                text: "Part 2".to_string(),
            },
        ]);
        let claude = convert_openai_to_claude_response(&resp, "claude-3", "req_123");
        match &claude.content[0] {
            ClaudeContentBlock::Text { text, .. } => assert_eq!(text, "Part 1 Part 2"),
            _ => panic!("Expected Text block"),
        }
    }

    // --- convert_openai_models_to_claude ---

    #[test]
    fn test_convert_models_response() {
        let openai_resp = OpenAIModelsResponse {
            object: "list".to_string(),
            data: vec![
                OpenAIModel {
                    id: "gpt-4".to_string(),
                    object: "model".to_string(),
                    created: 1700000000,
                    owned_by: "openai".to_string(),
                },
                OpenAIModel {
                    id: "gpt-3.5-turbo".to_string(),
                    object: "model".to_string(),
                    created: 1690000000,
                    owned_by: "openai".to_string(),
                },
            ],
        };
        let claude = convert_openai_models_to_claude(&openai_resp);
        assert_eq!(claude.data.len(), 2);
        assert_eq!(claude.data[0].id, "gpt-4");
        assert_eq!(claude.data[0].model_type, "model");
        assert_eq!(claude.data[0].display_name, "gpt-4");
        assert!(!claude.data[0].created_at.is_empty());
        assert_eq!(claude.first_id, Some("gpt-4".to_string()));
        assert_eq!(claude.last_id, Some("gpt-3.5-turbo".to_string()));
        assert!(!claude.has_more);
    }

    #[test]
    fn test_convert_empty_models() {
        let openai_resp = OpenAIModelsResponse {
            object: "list".to_string(),
            data: vec![],
        };
        let claude = convert_openai_models_to_claude(&openai_resp);
        assert!(claude.data.is_empty());
        assert!(claude.first_id.is_none());
        assert!(claude.last_id.is_none());
    }
}
