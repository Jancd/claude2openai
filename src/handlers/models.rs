use std::collections::HashMap;

use crate::converters::openai_to_claude::convert_openai_models_to_claude;
use crate::error::{handle_target_api_error, ProxyError};
use crate::types::openai::OpenAIModelsResponse;
use crate::validation::validate_models_request_params;

/// Handle GET /v1/models request.
pub async fn handle_models_request(
    client: &reqwest::Client,
    target_url: &str,
    auth_headers: &HashMap<String, String>,
    _request_id: &str,
    query_params: ModelsQueryParams,
) -> Result<axum::Json<crate::types::claude::ClaudeModelsResponse>, ProxyError> {
    // Validate parameters
    validate_models_request_params(query_params.limit)?;

    // Build target URL with query params
    let mut url = reqwest::Url::parse(target_url)
        .map_err(|e| ProxyError::Processing(format!("Invalid target URL: {e}")))?;

    if let Some(ref after_id) = query_params.after_id {
        url.query_pairs_mut().append_pair("after", after_id);
    }
    if let Some(ref before_id) = query_params.before_id {
        url.query_pairs_mut().append_pair("before", before_id);
    }
    if let Some(limit) = query_params.limit {
        url.query_pairs_mut()
            .append_pair("limit", &limit.to_string());
    }

    tracing::debug!("Upstream models request URL: {}", url);

    let mut req_builder = client.get(url).header("Content-Type", "application/json");
    for (key, value) in auth_headers {
        req_builder = req_builder.header(key, value);
    }

    let response = req_builder
        .send()
        .await
        .map_err(|e| ProxyError::Processing(format!("Failed to reach upstream: {e}")))?;

    if !response.status().is_success() {
        return Err(handle_target_api_error(
            response.status().as_u16(),
            "Models API",
        ));
    }

    let openai_resp: OpenAIModelsResponse = response
        .json()
        .await
        .map_err(|e| ProxyError::Processing(format!("Failed to parse models response: {e}")))?;

    let claude_resp = convert_openai_models_to_claude(&openai_resp);

    Ok(axum::Json(claude_resp))
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ModelsQueryParams {
    pub after_id: Option<String>,
    pub before_id: Option<String>,
    pub limit: Option<u32>,
}
