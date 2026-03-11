use std::sync::Arc;

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub port: u16,
    pub local_token_counting: bool,
    pub allowed_origins: Vec<String>,
    pub dev_mode: bool,
    pub allowed_hosts: Vec<String>,
    pub image_block_data_max_size: usize,
    pub fixed_route_target_url: Option<String>,
    pub fixed_route_path_prefix: String,
    pub log_level: String,
    pub tiktoken_model: String,
}

pub type SharedConfig = Arc<AppConfig>;

impl AppConfig {
    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8788);

        let local_token_counting = std::env::var("LOCAL_TOKEN_COUNTING")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let allowed_origins = std::env::var("ALLOWED_ORIGINS")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_else(|_| vec!["*".to_string()]);

        let dev_mode = std::env::var("DEV_MODE")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let allowed_hosts = std::env::var("ALLOWED_HOSTS")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_else(|_| vec!["127.0.0.1".to_string(), "localhost".to_string()]);

        let image_block_data_max_size = std::env::var("IMAGE_BLOCK_DATA_MAX_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_485_760);

        let fixed_route_target_url = std::env::var("FIXED_ROUTE_TARGET_URL").ok();

        let fixed_route_path_prefix = std::env::var("FIXED_ROUTE_PATH_PREFIX").unwrap_or_default();

        let log_level = std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

        let tiktoken_model =
            std::env::var("TIKTOKEN_MODEL").unwrap_or_else(|_| "cl100k_base".to_string());

        Self {
            port,
            local_token_counting,
            allowed_origins,
            dev_mode,
            allowed_hosts,
            image_block_data_max_size,
            fixed_route_target_url,
            fixed_route_path_prefix,
            log_level,
            tiktoken_model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        // Clear env vars that might interfere
        unsafe {
            std::env::remove_var("PORT");
            std::env::remove_var("LOCAL_TOKEN_COUNTING");
            std::env::remove_var("ALLOWED_ORIGINS");
            std::env::remove_var("DEV_MODE");
            std::env::remove_var("ALLOWED_HOSTS");
            std::env::remove_var("IMAGE_BLOCK_DATA_MAX_SIZE");
            std::env::remove_var("FIXED_ROUTE_TARGET_URL");
            std::env::remove_var("FIXED_ROUTE_PATH_PREFIX");
            std::env::remove_var("LOG_LEVEL");
            std::env::remove_var("TIKTOKEN_MODEL");
        }

        let config = AppConfig::from_env();
        assert_eq!(config.port, 8788);
        assert!(!config.local_token_counting);
        assert_eq!(config.allowed_origins, vec!["*".to_string()]);
        assert!(!config.dev_mode);
        assert!(config.allowed_hosts.contains(&"localhost".to_string()));
        assert!(config.allowed_hosts.contains(&"127.0.0.1".to_string()));
        assert_eq!(config.image_block_data_max_size, 10_485_760);
        assert!(config.fixed_route_target_url.is_none());
        assert_eq!(config.fixed_route_path_prefix, "");
        assert_eq!(config.log_level, "info");
        assert_eq!(config.tiktoken_model, "cl100k_base");
    }

    #[test]
    fn test_config_from_env() {
        unsafe {
            std::env::set_var("PORT", "3000");
            std::env::set_var("LOCAL_TOKEN_COUNTING", "true");
            std::env::set_var("DEV_MODE", "1");
            std::env::set_var("LOG_LEVEL", "debug");
        }

        let config = AppConfig::from_env();
        assert_eq!(config.port, 3000);
        assert!(config.local_token_counting);
        assert!(config.dev_mode);
        assert_eq!(config.log_level, "debug");

        // Clean up
        unsafe {
            std::env::remove_var("PORT");
            std::env::remove_var("LOCAL_TOKEN_COUNTING");
            std::env::remove_var("DEV_MODE");
            std::env::remove_var("LOG_LEVEL");
        }
    }

    #[test]
    fn test_config_allowed_origins_parsing() {
        unsafe {
            std::env::set_var("ALLOWED_ORIGINS", "http://localhost:3000, https://example.com");
        }
        let config = AppConfig::from_env();
        assert_eq!(config.allowed_origins.len(), 2);
        assert!(config.allowed_origins.contains(&"http://localhost:3000".to_string()));
        assert!(config.allowed_origins.contains(&"https://example.com".to_string()));
        unsafe {
            std::env::remove_var("ALLOWED_ORIGINS");
        }
    }

    #[test]
    fn test_config_invalid_port_uses_default() {
        unsafe {
            std::env::set_var("PORT", "invalid");
        }
        let config = AppConfig::from_env();
        assert_eq!(config.port, 8788);
        unsafe {
            std::env::remove_var("PORT");
        }
    }

    #[test]
    fn test_shared_config_type() {
        let config = AppConfig::from_env();
        let shared: SharedConfig = Arc::new(config);
        assert_eq!(shared.port, shared.port); // Just verify Arc works
    }
}
