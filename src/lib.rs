//! Jengine: a Rust inference runtime for compact LLMs.

/// Returns the project name.
pub fn name() -> &'static str {
    "jengine"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_name() {
        assert_eq!(name(), "jengine");
    }
}
