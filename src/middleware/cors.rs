use axum::http::{HeaderValue, Method};
use tower_http::cors::CorsLayer;

use crate::config::AppConfig;

/// Build a CORS layer based on configuration.
pub fn build_cors_layer(config: &AppConfig) -> CorsLayer {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            "content-type".parse().unwrap(),
            "authorization".parse().unwrap(),
            "x-api-key".parse().unwrap(),
            "anthropic-beta".parse().unwrap(),
        ])
        .max_age(std::time::Duration::from_secs(86400));

    if config.dev_mode || config.allowed_origins.contains(&"*".to_string()) {
        cors.allow_origin(tower_http::cors::Any)
    } else {
        let origins: Vec<HeaderValue> = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        cors.allow_origin(origins)
    }
}
