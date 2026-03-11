use std::collections::HashMap;

use axum::response::sse::{Event, KeepAlive, Sse};
use bytes::Bytes;
use futures::stream::{self, Stream, StreamExt};
use serde_json::json;

use crate::types::openai::OpenAIStreamChunk;

struct ToolCallState {
    id: String,
    name: String,
    #[allow(dead_code)]
    args: String,
    claude_index: usize,
    started: bool,
}

struct StreamTransformState {
    initialized: bool,
    buffer: String,
    message_id: String,
    model: String,
    tool_calls: HashMap<u32, ToolCallState>,
    content_block_index: usize,
    last_finish_reason: Option<String>,
}

fn make_sse_event(event_type: &str, data: &serde_json::Value) -> Event {
    Event::default()
        .event(event_type)
        .data(serde_json::to_string(data).unwrap_or_default())
}

/// Create a stream transformer that converts OpenAI SSE to Claude SSE format.
pub fn create_stream_transformer(
    upstream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: String,
    request_id: String,
) -> Sse<impl Stream<Item = Result<Event, anyhow::Error>>> {
    let state = StreamTransformState {
        initialized: false,
        buffer: String::new(),
        message_id: request_id,
        model,
        tool_calls: HashMap::new(),
        content_block_index: 0,
        last_finish_reason: None,
    };

    let stream = stream::unfold(
        (Box::pin(upstream) as std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>, state, false),
        |(mut upstream, mut state, done)| async move {
            if done {
                return None;
            }

            match upstream.next().await {
                Some(Ok(chunk)) => {
                    let chunk_text = String::from_utf8_lossy(&chunk);
                    let mut events: Vec<Result<Event, anyhow::Error>> = Vec::new();

                    if !state.initialized {
                        // Send message_start
                        events.push(Ok(make_sse_event(
                            "message_start",
                            &json!({
                                "type": "message_start",
                                "message": {
                                    "id": state.message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "model": state.model,
                                    "content": [],
                                    "stop_reason": null,
                                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                                }
                            }),
                        )));
                        // Send content_block_start for first text block
                        events.push(Ok(make_sse_event(
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": { "type": "text", "text": "" }
                            }),
                        )));
                        state.initialized = true;
                    }

                    state.buffer.push_str(&chunk_text);
                    let lines: Vec<String> = state.buffer.split('\n').map(|s| s.to_string()).collect();
                    let last = lines.last().cloned().unwrap_or_default();
                    let complete_lines: Vec<String> = lines[..lines.len().saturating_sub(1)].to_vec();
                    state.buffer = last;

                    for line in &complete_lines {
                        if !line.starts_with("data: ") {
                            continue;
                        }
                        let data = &line[6..];

                        if data.trim() == "[DONE]" {
                            // Send content_block_stop for text block
                            events.push(Ok(make_sse_event(
                                "content_block_stop",
                                &json!({ "type": "content_block_stop", "index": 0 }),
                            )));

                            // Send content_block_stop for all started tool calls
                            let mut tc_indices: Vec<u32> =
                                state.tool_calls.keys().cloned().collect();
                            tc_indices.sort();
                            for idx in tc_indices {
                                if let Some(tc) = state.tool_calls.get(&idx) {
                                    if tc.started {
                                        events.push(Ok(make_sse_event(
                                            "content_block_stop",
                                            &json!({
                                                "type": "content_block_stop",
                                                "index": tc.claude_index
                                            }),
                                        )));
                                    }
                                }
                            }

                            // Determine final stop reason
                            let final_stop_reason = match state.last_finish_reason.as_deref() {
                                Some("tool_calls") => "tool_use",
                                Some("length") => "max_tokens",
                                Some("content_filter") => "content_filter",
                                _ => "end_turn",
                            };

                            // Send message_delta
                            events.push(Ok(make_sse_event(
                                "message_delta",
                                &json!({
                                    "type": "message_delta",
                                    "delta": { "stop_reason": final_stop_reason, "stop_sequence": null },
                                    "usage": { "output_tokens": 0 }
                                }),
                            )));

                            // Send message_stop
                            events.push(Ok(make_sse_event(
                                "message_stop",
                                &json!({ "type": "message_stop" }),
                            )));

                            let event_stream = stream::iter(events);
                            return Some((event_stream, (upstream, state, true)));
                        }

                        // Parse OpenAI chunk
                        let openai_chunk: OpenAIStreamChunk = match serde_json::from_str(data) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };

                        if openai_chunk.choices.is_empty() {
                            continue;
                        }

                        let choice = &openai_chunk.choices[0];
                        let delta = &choice.delta;

                        // Track finish_reason
                        if let Some(ref reason) = choice.finish_reason {
                            state.last_finish_reason = Some(reason.clone());
                        }

                        // Handle text content delta
                        if let Some(ref content) = delta.content {
                            events.push(Ok(make_sse_event(
                                "content_block_delta",
                                &json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": { "type": "text_delta", "text": content }
                                }),
                            )));
                        }

                        // Handle tool call deltas
                        if let Some(ref tool_calls) = delta.tool_calls {
                            for tc_delta in tool_calls {
                                let index = tc_delta.index;
                                let entry =
                                    state.tool_calls.entry(index).or_insert_with(|| ToolCallState {
                                        id: String::new(),
                                        name: String::new(),
                                        args: String::new(),
                                        claude_index: 0,
                                        started: false,
                                    });

                                if let Some(ref id) = tc_delta.id {
                                    entry.id = id.clone();
                                }
                                if let Some(ref func) = tc_delta.function {
                                    if let Some(ref name) = func.name {
                                        entry.name = name.clone();
                                    }
                                    if let Some(ref args) = func.arguments {
                                        entry.args.push_str(args);
                                    }
                                }

                                // Start new tool_use block when we have id and name
                                if !entry.id.is_empty()
                                    && !entry.name.is_empty()
                                    && !entry.started
                                {
                                    state.content_block_index += 1;
                                    let claude_idx = state.content_block_index;
                                    entry.claude_index = claude_idx;
                                    entry.started = true;

                                    events.push(Ok(make_sse_event(
                                        "content_block_start",
                                        &json!({
                                            "type": "content_block_start",
                                            "index": claude_idx,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": entry.id,
                                                "name": entry.name,
                                                "input": {}
                                            }
                                        }),
                                    )));
                                }

                                // Send input_json_delta for tool arguments
                                if entry.started {
                                    if let Some(ref func) = tc_delta.function {
                                        if let Some(ref args) = func.arguments {
                                            events.push(Ok(make_sse_event(
                                                "content_block_delta",
                                                &json!({
                                                    "type": "content_block_delta",
                                                    "index": entry.claude_index,
                                                    "delta": {
                                                        "type": "input_json_delta",
                                                        "partial_json": args
                                                    }
                                                }),
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let event_stream = stream::iter(events);
                    Some((event_stream, (upstream, state, false)))
                }
                Some(Err(e)) => {
                    let events = vec![Err(anyhow::anyhow!("Upstream error: {}", e))];
                    let event_stream = stream::iter(events);
                    Some((event_stream, (upstream, state, true)))
                }
                None => {
                    // Stream ended without [DONE] — send cleanup events
                    let mut events: Vec<Result<Event, anyhow::Error>> = Vec::new();
                    if state.initialized {
                        events.push(Ok(make_sse_event(
                            "message_delta",
                            &json!({
                                "type": "message_delta",
                                "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                                "usage": { "output_tokens": 0 }
                            }),
                        )));
                        events.push(Ok(make_sse_event(
                            "message_stop",
                            &json!({ "type": "message_stop" }),
                        )));
                    }
                    let event_stream = stream::iter(events);
                    Some((event_stream, (upstream, state, true)))
                }
            }
        },
    )
    .flatten();

    Sse::new(stream).keep_alive(KeepAlive::default())
}
