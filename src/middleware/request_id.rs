use uuid::Uuid;

/// Generate a unique request ID.
pub fn generate_request_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let rand_part = &Uuid::new_v4().to_string()[..9];
    format!("req_{ts}_{rand_part}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();
        assert!(id.starts_with("req_"));
        let parts: Vec<&str> = id.splitn(3, '_').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "req");
        // timestamp part should be numeric
        assert!(parts[1].parse::<u128>().is_ok());
        // random part should be 9 chars
        assert_eq!(parts[2].len(), 9);
    }

    #[test]
    fn test_generate_request_id_unique() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();
        assert_ne!(id1, id2);
    }
}
