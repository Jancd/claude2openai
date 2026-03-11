use std::collections::HashMap;

use crate::config::AppConfig;
use crate::converters::claude_to_openai::convert_claude_token_counting_to_openai;
use crate::error::{handle_target_api_error, ProxyError};
use crate::token_counting::{count_claude_request_tokens, get_tokenizer};
use crate::types::claude::{ClaudeTokenCountingRequest, ClaudeTokenCountingResponse};
use crate::types::openai::OpenAIResponse;
use crate::validation::{validate_auth_headers, validate_claude_token_counting_request};

/// Handle POST /v1/messages/count_tokens request.
pub async fn handle_token_counting_request(
    client: &reqwest::Client,
    mut target_url: String,
    auth_headers: &HashMap<String, String>,
    request_id: &str,
    config: &AppConfig,
    claude_request: ClaudeTokenCountingRequest,
) -> Result<axum::Json<ClaudeTokenCountingResponse>, ProxyError> {
    // Validate request
    validate_claude_token_counting_request(&claude_request, config.image_block_data_max_size)?;
    validate_auth_headers(auth_headers)?;

    if config.local_token_counting {
        return handle_local_token_counting(&claude_request, request_id, config);
    }

    // API-based token counting
    let openai_request =
        convert_claude_token_counting_to_openai(&claude_request, &claude_request.model);

    // Replace endpoint path
    if target_url.contains("v1/messages/count_tokens") {
        target_url = target_url.replace("v1/messages/count_tokens", "v1/chat/completions");
    }

    tracing::debug!("Upstream token counting request url: {}", target_url);

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
            "Token Counting API",
        ));
    }

    let openai_resp: OpenAIResponse = response
        .json()
        .await
        .map_err(|e| ProxyError::Processing(format!("Failed to parse response: {e}")))?;

    let input_tokens = openai_resp.usage.as_ref().map_or(0, |u| u.prompt_tokens);

    Ok(axum::Json(ClaudeTokenCountingResponse {
        response_type: "token_count".to_string(),
        input_tokens,
    }))
}

/// Handle token counting using local estimation or tiktoken.
fn handle_local_token_counting(
    claude_request: &ClaudeTokenCountingRequest,
    request_id: &str,
    config: &AppConfig,
) -> Result<axum::Json<ClaudeTokenCountingResponse>, ProxyError> {
    let tokenizer = get_tokenizer(&config.tiktoken_model);

    let input_tokens = count_claude_request_tokens(claude_request, tokenizer) as u32;

    tracing::debug!(
        request_id = request_id,
        "Local token count: {}",
        input_tokens
    );

    Ok(axum::Json(ClaudeTokenCountingResponse {
        response_type: "token_count".to_string(),
        input_tokens,
    }))
}
