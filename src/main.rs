mod beta_features;
mod config;
mod converters;
mod error;
mod handlers;
mod middleware;
mod routing;
mod thinking;
mod token_counting;
mod types;
mod validation;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Query, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::json;
use tower_http::limit::RequestBodyLimitLayer;

use config::{AppConfig, SharedConfig};
use error::ProxyError;
use handlers::models::ModelsQueryParams;
use middleware::request_id::generate_request_id;
use routing::{
    build_target_url, extract_auth_headers, get_handler_type, is_host_allowed, parse_dynamic_route,
    HandlerType,
};

/// Shared application state.
#[derive(Clone)]
struct AppState {
    config: SharedConfig,
    client: reqwest::Client,
}

#[tokio::main]
async fn main() {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Load config
    let config = AppConfig::from_env();

    // Initialize tracing
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.log_level));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let port = config.port;
    let shared_config: SharedConfig = Arc::new(config);

    // Build HTTP client with connection pooling
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(20)
        .build()
        .expect("Failed to build HTTP client");

    let state = AppState {
        config: shared_config.clone(),
        client,
    };

    // Build CORS layer
    let cors_layer = middleware::cors::build_cors_layer(&shared_config);

    let app = Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/", get(health_check))
        // Favicon
        .route("/favicon.ico", get(|| async { StatusCode::NO_CONTENT }))
        // Fixed routes
        .route("/v1/messages", post(handle_fixed_messages))
        .route(
            "/v1/messages/count_tokens",
            post(handle_fixed_token_counting),
        )
        .route("/v1/models", get(handle_fixed_models))
        // Dynamic route fallback
        .fallback(handle_dynamic_route)
        // Middleware
        .layer(cors_layer)
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("Failed to bind");
    tracing::info!("Server running on http://0.0.0.0:{port}");
    axum::serve(listener, app).await.expect("Server failed");
}

// --- Health Check ---

async fn health_check(State(state): State<AppState>) -> Response {
    let target_url = state
        .config
        .fixed_route_target_url
        .as_deref()
        .unwrap_or("https://api.openai.com");
    let url = format!("{target_url}/v1/models");

    match state.client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body: serde_json::Value = resp.json().await.unwrap_or(json!({}));
            let model_count = body
                .get("data")
                .and_then(|d| d.as_array())
                .map_or(0, |a| a.len());
            Json(json!({ "status": "ok", "models": model_count })).into_response()
        }
        _ => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "No models Found." })),
        )
            .into_response(),
    }
}

// --- Fixed Route Handlers ---

async fn handle_fixed_messages(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<types::claude::ClaudeMessagesRequest>,
) -> Result<Response, ProxyError> {
    let request_id = generate_request_id();
    let auth_headers = extract_auth_headers(&headers);

    let base_url = state
        .config
        .fixed_route_target_url
        .as_deref()
        .unwrap_or("https://api.example.com");
    let prefix = &state.config.fixed_route_path_prefix;
    let target_url = format!("{base_url}{prefix}/v1/chat/completions");

    handlers::messages::handle_messages_request(
        &state.client,
        target_url,
        &auth_headers,
        &request_id,
        None,
        state.config.image_block_data_max_size,
        body,
    )
    .await
}

async fn handle_fixed_token_counting(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<types::claude::ClaudeTokenCountingRequest>,
) -> Result<Json<types::claude::ClaudeTokenCountingResponse>, ProxyError> {
    let request_id = generate_request_id();
    let auth_headers = extract_auth_headers(&headers);

    let base_url = state
        .config
        .fixed_route_target_url
        .as_deref()
        .unwrap_or("https://api.example.com");
    let prefix = &state.config.fixed_route_path_prefix;
    let target_url = format!("{base_url}{prefix}/v1/messages/count_tokens");

    handlers::token_counting::handle_token_counting_request(
        &state.client,
        target_url,
        &auth_headers,
        &request_id,
        &state.config,
        body,
    )
    .await
}

async fn handle_fixed_models(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(params): Query<ModelsQueryParams>,
) -> Result<Json<types::claude::ClaudeModelsResponse>, ProxyError> {
    let request_id = generate_request_id();
    let auth_headers = extract_auth_headers(&headers);

    let base_url = state
        .config
        .fixed_route_target_url
        .as_deref()
        .unwrap_or("https://api.example.com");
    let prefix = &state.config.fixed_route_path_prefix;
    let target_url = format!("{base_url}{prefix}/v1/models");

    handlers::models::handle_models_request(
        &state.client,
        &target_url,
        &auth_headers,
        &request_id,
        params,
    )
    .await
}

// --- Dynamic Route Handler ---

async fn handle_dynamic_route(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    request: Request<Body>,
) -> Result<Response, ProxyError> {
    let path = request.uri().path().to_string();

    // Only handle dynamic routes
    if !path.starts_with("/http/") && !path.starts_with("/https/") {
        return Err(ProxyError::Validation(format!("Unknown route: {path}")));
    }

    let parsed = parse_dynamic_route(&path)?;
    let host = parsed
        .target_config
        .target_url
        .replace("https://", "")
        .replace("http://", "");

    // SSRF check
    if !is_host_allowed(&host, &state.config.allowed_hosts) {
        return Err(ProxyError::SsrfBlocked(format!("Host not allowed: {host}")));
    }

    let request_id = generate_request_id();
    let auth_headers = extract_auth_headers(&headers);
    let handler_type = get_handler_type(&parsed.claude_endpoint)?;
    let target_url = build_target_url(
        &parsed.target_config,
        &parsed.claude_endpoint,
        parsed.model_id.as_deref(),
    );

    match handler_type {
        HandlerType::Models => {
            // Parse query params from the original request
            let query_string = request.uri().query().unwrap_or("");
            let params: ModelsQueryParams =
                serde_urlencoded::from_str(query_string).unwrap_or_default();

            let result = handlers::models::handle_models_request(
                &state.client,
                &target_url,
                &auth_headers,
                &request_id,
                params,
            )
            .await?;
            Ok(result.into_response())
        }
        HandlerType::Messages => {
            let body = axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024)
                .await
                .map_err(|e| ProxyError::Validation(format!("Failed to read body: {e}")))?;
            let claude_request: types::claude::ClaudeMessagesRequest =
                serde_json::from_slice(&body)
                    .map_err(|e| ProxyError::Validation(format!("Invalid JSON: {e}")))?;

            handlers::messages::handle_messages_request(
                &state.client,
                target_url,
                &auth_headers,
                &request_id,
                parsed.model_id.as_deref(),
                state.config.image_block_data_max_size,
                claude_request,
            )
            .await
        }
        HandlerType::TokenCounting => {
            let body = axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024)
                .await
                .map_err(|e| ProxyError::Validation(format!("Failed to read body: {e}")))?;
            let claude_request: types::claude::ClaudeTokenCountingRequest =
                serde_json::from_slice(&body)
                    .map_err(|e| ProxyError::Validation(format!("Invalid JSON: {e}")))?;

            let result = handlers::token_counting::handle_token_counting_request(
                &state.client,
                target_url,
                &auth_headers,
                &request_id,
                &state.config,
                claude_request,
            )
            .await?;
            Ok(result.into_response())
        }
    }
}
