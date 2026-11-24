use std::fmt;
use std::path::PathBuf;
use uuid::Uuid;

/// Wrapper around an online-mode Minecraft access token that never prints secret contents
/// in `Debug` or `Display` output.
#[derive(Clone, PartialEq, Eq)]
pub struct RedactedToken(String);

impl RedactedToken {
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for RedactedToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl fmt::Display for RedactedToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

/// Authenticated online-mode player session configuration (to be supplied by `solara-launcher`).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub username: String,
    pub uuid: Uuid,
    pub access_token: RedactedToken,
    pub assets_dir: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_and_display_never_leak_access_token() {
        let cfg = SessionConfig {
            username: "SolaraPlayer".into(),
            uuid: Uuid::nil(),
            access_token: RedactedToken::new("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret"),
            assets_dir: PathBuf::from("/tmp/assets"),
        };
        let formatted = format!("{cfg:?}");
        assert!(!formatted.contains("eyJhbGci"));
        assert!(formatted.contains("[REDACTED]"));
    }
}
