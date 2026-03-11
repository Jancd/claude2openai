use std::collections::HashMap;

use axum::response::{IntoResponse, Response};

use crate::converters::claude_to_openai::convert_claude_to_openai_request;
use crate::converters::openai_to_claude::convert_openai_to_claude_response;
use crate::converters::streaming::create_stream_transformer;
use crate::error::{handle_target_api_error, ProxyError};
use crate::types::claude::ClaudeMessagesRequest;
use crate::types::openai::OpenAIResponse;
use crate::validation::{validate_auth_headers, validate_claude_messages_request};

/// Handle POST /v1/messages request.
pub async fn handle_messages_request(
    client: &reqwest::Client,
    mut target_url: String,
    auth_headers: &HashMap<String, String>,
    request_id: &str,
    model_id: Option<&str>,
    max_image_data_size: usize,
    claude_request: ClaudeMessagesRequest,
) -> Result<Response, ProxyError> {
    // Validate request
    validate_claude_messages_request(&claude_request, model_id, max_image_data_size)?;
    validate_auth_headers(auth_headers)?;

    // Use model from URL if provided, otherwise from request body
    let target_model_id = model_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| claude_request.model.clone());

    // Convert Claude request to OpenAI format
    let openai_request = convert_claude_to_openai_request(&claude_request, &target_model_id);

    // Replace endpoint path
    if target_url.contains("v1/messages") {
        target_url = target_url.replace("v1/messages", "v1/chat/completions");
    }

    let is_streaming = claude_request.stream.unwrap_or(false);

    tracing::debug!("Upstream request url: {}", target_url);
    tracing::debug!("Is streaming: {}", is_streaming);

    let mut req_builder = client
        .post(&target_url)
        .header("Content-Type", "application/json");

    for (key, value) in auth_headers {
        req_builder = req_builder.header(key, value);
    }

    let response = req_builder
        .json(&openai_request)
        .send()
        .await
        .map_err(|e| ProxyError::Processing(format!("Failed to reach upstream: {e}")))?;

    if !response.status().is_success() {
        return Err(handle_target_api_error(
            response.status().as_u16(),
            "Messages API",
        ));
    }

    if is_streaming {
        handle_streaming_response(response, &target_model_id, request_id).await
    } else {
        handle_non_streaming_response(response, &target_model_id, request_id).await
    }
}

/// Handle non-streaming response: parse OpenAI JSON and convert to Claude format.
async fn handle_non_streaming_response(
    response: reqwest::Response,
    model: &str,
    request_id: &str,
) -> Result<Response, ProxyError> {
    let openai_resp: OpenAIResponse = response
        .json()
        .await
        .map_err(|e| ProxyError::Processing(format!("Failed to parse response: {e}")))?;

    let claude_resp = convert_openai_to_claude_response(&openai_resp, model, request_id);

    let mut response = axum::Json(claude_resp).into_response();
    if let Ok(val) = request_id.parse() {
        response.headers_mut().insert("x-request-id", val);
    }
    Ok(response)
}

/// Handle streaming response: pipe upstream SSE through the stream transformer.
async fn handle_streaming_response(
    response: reqwest::Response,
    model: &str,
    request_id: &str,
) -> Result<Response, ProxyError> {
    let byte_stream = response.bytes_stream();
    let sse = create_stream_transformer(byte_stream, model.to_string(), request_id.to_string());

    let mut response = sse.into_response();
    if let Ok(val) = request_id.parse() {
        response.headers_mut().insert("x-request-id", val);
    }
    if let Ok(val) = "no-cache".parse() {
        response.headers_mut().insert("Cache-Control", val);
    }
    Ok(response)
}
