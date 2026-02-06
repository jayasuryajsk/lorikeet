#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use crate::verify::detect_suggestions;

    #[test]
    fn node_suggestions_prefer_existing_scripts() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{
  "name": "x",
  "scripts": {
    "lint": "eslint .",
    "typecheck": "tsc -p ."
  }
}"#,
        )
        .unwrap();
        fs::write(dir.path().join("pnpm-lock.yaml"), "lock").unwrap();

        let s = detect_suggestions(dir.path());
        let cmds: Vec<String> = s.iter().map(|x| x.command.clone()).collect();

        assert!(cmds.iter().any(|c| c == "pnpm run lint"));
        assert!(cmds.iter().any(|c| c == "pnpm run typecheck"));
        // Build shouldn't be confidently suggested when not present.
        assert!(!cmds.iter().any(|c| c == "pnpm run build"));
    }
}
