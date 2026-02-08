use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::session::SessionStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointBackend {
    Git,
    Snapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub id: String,
    pub name: Option<String>,
    pub created_at_unix: i64,
    pub workspace_root: String,
    pub project_id: String,
    pub backend: CheckpointBackend,
    pub session_id: String,
    pub session_event_count: usize,
    pub git_head: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileEntry {
    path: String,
    size: u64,
    #[serde(default)]
    mode: Option<u32>,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    files: Vec<FileEntry>,
}

pub fn checkpoints_base_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home dir"))?;
    Ok(home.join(".lorikeet").join("checkpoints"))
}

pub fn checkpoints_dir_for_workspace(workspace_root: &Path) -> Result<PathBuf> {
    Ok(checkpoints_base_dir()?.join(project_id(workspace_root)))
}

pub fn checkpoint_dir(workspace_root: &Path, id: &str) -> Result<PathBuf> {
    Ok(checkpoints_dir_for_workspace(workspace_root)?.join(id))
}

pub fn load_checkpoint_meta(workspace_root: &Path, id: &str) -> Result<CheckpointMeta> {
    let dir = checkpoint_dir(workspace_root, id)?;
    let meta_path = dir.join("meta.json");
    let data = fs::read_to_string(&meta_path)
        .with_context(|| format!("Failed to read {}", meta_path.display()))?;
    let meta: CheckpointMeta = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse {}", meta_path.display()))?;
    Ok(meta)
}

pub fn list_checkpoints(workspace_root: &Path, limit: usize) -> Result<Vec<CheckpointMeta>> {
    let dir = checkpoints_dir_for_workspace(workspace_root)?;
    let mut metas = Vec::new();
    let Ok(rd) = fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    for ent in rd.flatten() {
        let p = ent.path();
        if !p.is_dir() {
            continue;
        }
        let meta_path = p.join("meta.json");
        let Ok(data) = fs::read_to_string(&meta_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<CheckpointMeta>(&data) else {
            continue;
        };
        metas.push(meta);
    }
    metas.sort_by_key(|m| std::cmp::Reverse(m.created_at_unix));
    metas.truncate(limit);
    Ok(metas)
}

pub fn create_checkpoint(
    workspace_root: &Path,
    session: &SessionStore,
    name: Option<String>,
) -> Result<CheckpointMeta> {
    let workspace_root =
        fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let backend = if is_git_worktree(&workspace_root) {
        CheckpointBackend::Git
    } else {
        CheckpointBackend::Snapshot
    };

    let id = new_checkpoint_id();
    let dir = checkpoint_dir(&workspace_root, &id)?;
    fs::create_dir_all(&dir).with_context(|| format!("Failed to create {}", dir.display()))?;

    let created_at_unix = unix_ts();
    let project = project_id(&workspace_root);

    // Record the checkpoint marker event first, then store the exact line count after it.
    session.record_checkpoint(&id, name.as_deref());
    let session_event_count = session.count_events_lines()?;

    let mut git_head: Option<String> = None;
    match backend {
        CheckpointBackend::Git => {
            git_head = Some(
                run_git(&workspace_root, &["rev-parse", "HEAD"])?
                    .trim()
                    .to_string(),
            );
            git_capture(&workspace_root, &dir)?;
        }
        CheckpointBackend::Snapshot => {
            snapshot_capture(&workspace_root, &dir)?;
        }
    }

    let meta = CheckpointMeta {
        id: id.clone(),
        name,
        created_at_unix,
        workspace_root: workspace_root.display().to_string(),
        project_id: project,
        backend,
        session_id: session.session_id.clone(),
        session_event_count,
        git_head,
        notes: None,
    };
    let meta_path = dir.join("meta.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("Failed to write {}", meta_path.display()))?;
    Ok(meta)
}

pub fn restore_checkpoint(
    workspace_root: &Path,
    session: &SessionStore,
    meta: &CheckpointMeta,
) -> Result<()> {
    let workspace_root =
        fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let canon_meta_root = fs::canonicalize(Path::new(&meta.workspace_root))
        .unwrap_or_else(|_| PathBuf::from(&meta.workspace_root));
    if workspace_root != canon_meta_root {
        return Err(anyhow!(
            "Checkpoint workspace mismatch: meta={} current={}",
            canon_meta_root.display(),
            workspace_root.display()
        ));
    }

    // Pre-restore safety checkpoint. Note: its session rewind may not be possible after truncation,
    // but the file snapshot is still a reliable escape hatch.
    let _ = create_checkpoint(
        &workspace_root,
        session,
        Some(format!("pre-restore {}", meta.id)),
    );

    let dir = checkpoint_dir(&workspace_root, &meta.id)?;
    match meta.backend {
        CheckpointBackend::Git => git_restore(&workspace_root, &dir, meta)?,
        CheckpointBackend::Snapshot => snapshot_restore(&workspace_root, &dir)?,
    }

    Ok(())
}

pub fn checkpoint_diff_summary(workspace_root: &Path, meta: &CheckpointMeta) -> Result<String> {
    let workspace_root =
        fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let dir = checkpoint_dir(&workspace_root, &meta.id)?;
    match meta.backend {
        CheckpointBackend::Git => {
            let staged = dir.join("staged.patch");
            let unstaged = dir.join("unstaged.patch");
            let mut out = String::new();
            if let Some(head) = &meta.git_head {
                out.push_str(&format!("git head: {}\n", head));
            }
            out.push_str("staged:\n");
            out.push_str(
                &git_patch_numstat(&workspace_root, &staged)
                    .unwrap_or_else(|_| "(unavailable)\n".to_string()),
            );
            out.push_str("unstaged:\n");
            out.push_str(
                &git_patch_numstat(&workspace_root, &unstaged)
                    .unwrap_or_else(|_| "(unavailable)\n".to_string()),
            );
            Ok(out.trim_end().to_string())
        }
        CheckpointBackend::Snapshot => {
            let manifest_path = dir.join("manifest.json");
            let data = fs::read_to_string(&manifest_path)
                .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
            let manifest: Manifest = serde_json::from_str(&data)
                .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;

            let mut expected: BTreeMap<String, String> = BTreeMap::new();
            for f in manifest.files {
                expected.insert(f.path, f.sha256);
            }

            let mut cur: BTreeMap<String, String> = BTreeMap::new();
            for rel in expected.keys() {
                let p = workspace_root.join(rel);
                if p.exists() {
                    cur.insert(rel.clone(), sha256_hex(&p)?);
                }
            }

            let mut added = 0usize;
            let mut removed = 0usize;
            let mut changed = 0usize;
            for (k, v) in expected.iter() {
                match cur.get(k) {
                    None => removed += 1,
                    Some(h) if h != v => changed += 1,
                    _ => {}
                }
            }
            // Count extra current files (within included set).
            let cur_files = list_included_files(&workspace_root)?;
            for rel in cur_files {
                if !expected.contains_key(&rel) {
                    added += 1;
                }
            }

            Ok(format!(
                "snapshot diff (approx): +{} -{} ~{}",
                added, removed, changed
            ))
        }
    }
}

pub fn truncate_session_to(store: &SessionStore, event_count: usize) -> Result<()> {
    Ok(store.truncate_to_lines(event_count)?)
}

fn git_capture(workspace_root: &Path, dir: &Path) -> Result<()> {
    let staged = dir.join("staged.patch");
    let unstaged = dir.join("unstaged.patch");

    run_git_to_file(workspace_root, &["diff", "--binary", "--staged"], &staged)?;
    run_git_to_file(workspace_root, &["diff", "--binary"], &unstaged)?;

    let untracked = run_git_bytes(
        workspace_root,
        &["ls-files", "--others", "--exclude-standard", "-z"],
    )?;
    let files = split_nul(&untracked);

    let mut entries = Vec::new();
    for rel in files {
        if rel.trim().is_empty() {
            continue;
        }
        let rel_path = Path::new(&rel);
        if rel_path.is_absolute()
            || rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            continue;
        }
        let src = workspace_root.join(rel_path);
        if !src.is_file() {
            continue;
        }
        let dst = dir.join("untracked").join(rel_path);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst)?;

        let md = fs::metadata(&src)?;
        let mode = file_mode(&md);
        let sha = sha256_hex(&src)?;
        entries.push(FileEntry {
            path: rel.to_string(),
            size: md.len(),
            mode,
            sha256: sha,
        });
    }

    let manifest = Manifest { files: entries };
    fs::write(
        dir.join("untracked_manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

fn git_restore(workspace_root: &Path, dir: &Path, meta: &CheckpointMeta) -> Result<()> {
    if !is_git_worktree(workspace_root) {
        return Err(anyhow!("Not a git worktree"));
    }
    let head = meta
        .git_head
        .as_ref()
        .ok_or_else(|| anyhow!("Missing git_head in checkpoint meta"))?;

    // Reset tracked state to the checkpoint base.
    let restore_ok = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(["restore", "--source", head, "--staged", "--worktree", ":/"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !restore_ok {
        run_git_status_ok(workspace_root, &["reset", "--hard", head])?;
    }

    // Remove untracked.
    let _ = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(["clean", "-fd"])
        .status();

    // Restore checkpoint untracked.
    let untracked_dir = dir.join("untracked");
    if untracked_dir.exists() {
        copy_tree(&untracked_dir, workspace_root, Some(workspace_root))?;
    }

    // Apply patches.
    let staged = dir.join("staged.patch");
    if staged.exists() && fs::metadata(&staged).map(|m| m.len() > 0).unwrap_or(false) {
        run_git_status_ok(
            workspace_root,
            &[
                "apply",
                "--binary",
                "--index",
                staged.to_string_lossy().as_ref(),
            ],
        )?;
    }
    let unstaged = dir.join("unstaged.patch");
    if unstaged.exists()
        && fs::metadata(&unstaged)
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    {
        run_git_status_ok(
            workspace_root,
            &["apply", "--binary", unstaged.to_string_lossy().as_ref()],
        )?;
    }

    Ok(())
}

fn snapshot_capture(workspace_root: &Path, dir: &Path) -> Result<()> {
    let snap_dir = dir.join("snapshot");
    fs::create_dir_all(&snap_dir)?;

    let mut entries = Vec::new();
    for rel in list_included_files(workspace_root)? {
        let src = workspace_root.join(&rel);
        if !src.is_file() {
            continue;
        }
        let dst = snap_dir.join(&rel);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst)?;

        let md = fs::metadata(&src)?;
        let sha = sha256_hex(&src)?;
        entries.push(FileEntry {
            path: rel,
            size: md.len(),
            mode: file_mode(&md),
            sha256: sha,
        });
    }

    let manifest = Manifest { files: entries };
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

fn snapshot_restore(workspace_root: &Path, dir: &Path) -> Result<()> {
    let manifest_path = dir.join("manifest.json");
    let data = fs::read_to_string(&manifest_path)
        .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;
    let expected: HashSet<String> = manifest.files.iter().map(|f| f.path.clone()).collect();

    // Delete files not in manifest, but only within included (non-excluded) set.
    for rel in list_included_files(workspace_root)? {
        if !expected.contains(&rel) {
            let p = workspace_root.join(&rel);
            let _ = fs::remove_file(&p);
        }
    }

    // Restore snapshot.
    let snap_dir = dir.join("snapshot");
    copy_tree(&snap_dir, workspace_root, Some(workspace_root))?;

    // Best-effort restore modes.
    for f in manifest.files {
        if let Some(mode) = f.mode {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let p = workspace_root.join(&f.path);
                if let Ok(md) = fs::metadata(&p) {
                    let mut perms = md.permissions();
                    perms.set_mode(mode);
                    let _ = fs::set_permissions(&p, perms);
                }
            }
        }
    }

    Ok(())
}

fn is_git_worktree(workspace_root: &Path) -> bool {
    std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_git(workspace_root: &Path, args: &[&str]) -> Result<String> {
    let out = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(args)
        .output()
        .with_context(|| format!("Failed to run git {}", args.join(" ")))?;
    if !out.status.success() {
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run_git_bytes(workspace_root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let out = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(args)
        .output()
        .with_context(|| format!("Failed to run git {}", args.join(" ")))?;
    if !out.status.success() {
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

fn run_git_to_file(workspace_root: &Path, args: &[&str], out_path: &Path) -> Result<()> {
    let out = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(args)
        .output()
        .with_context(|| format!("Failed to run git {}", args.join(" ")))?;
    if !out.status.success() {
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    fs::write(out_path, out.stdout)
        .with_context(|| format!("Failed to write {}", out_path.display()))?;
    Ok(())
}

fn run_git_status_ok(workspace_root: &Path, args: &[&str]) -> Result<()> {
    let status = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(args)
        .status()
        .with_context(|| format!("Failed to run git {}", args.join(" ")))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("git {} failed", args.join(" ")))
    }
}

fn git_patch_numstat(workspace_root: &Path, patch_path: &Path) -> Result<String> {
    let out = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args([
            "apply",
            "--numstat",
            "--summary",
            patch_path.to_string_lossy().as_ref(),
        ])
        .output()
        .with_context(|| "Failed to run git apply --numstat")?;
    if !out.status.success() {
        return Err(anyhow!("git apply --numstat failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn split_nul(bytes: &[u8]) -> Vec<String> {
    bytes
        .split(|b| *b == 0)
        .filter_map(|s| {
            if s.is_empty() {
                None
            } else {
                Some(String::from_utf8_lossy(s).to_string())
            }
        })
        .collect()
}

fn copy_tree(src_root: &Path, dst_root: &Path, workspace_root: Option<&Path>) -> Result<()> {
    for ent in WalkDir::new(src_root) {
        let ent = ent?;
        if ent.file_type().is_dir() {
            continue;
        }
        let src = ent.path();
        let rel = src.strip_prefix(src_root).unwrap_or(src);
        if rel
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            continue;
        }
        let dst = dst_root.join(rel);

        if let Some(ws) = workspace_root {
            // Defensive: ensure destination is within workspace root.
            let canon = fs::canonicalize(dst.parent().unwrap_or(dst_root))
                .unwrap_or_else(|_| dst.parent().unwrap_or(dst_root).to_path_buf());
            let ws_canon = fs::canonicalize(ws).unwrap_or_else(|_| ws.to_path_buf());
            if !canon.starts_with(&ws_canon) {
                return Err(anyhow!(
                    "Refusing to write outside workspace: {}",
                    dst.display()
                ));
            }
        }

        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, &dst)?;
    }
    Ok(())
}

fn list_included_files(workspace_root: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for ent in WalkDir::new(workspace_root).follow_links(false) {
        let ent = ent?;
        let p = ent.path();
        let rel = match p.strip_prefix(workspace_root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if rel.as_os_str().is_empty() {
            continue;
        }
        if is_excluded(rel) {
            if ent.file_type().is_dir() {
                // walkdir doesn't support pruning without filter_entry; keep it simple by skipping
                // based on path checks below.
            }
            continue;
        }
        if ent.file_type().is_file() {
            out.push(rel.to_string_lossy().to_string());
        }
    }
    Ok(out)
}

fn is_excluded(rel: &Path) -> bool {
    // Exclude common heavy/build dirs and lorikeet internals.
    let excluded = [
        ".git",
        "target",
        "node_modules",
        "dist",
        "build",
        ".lorikeet",
    ];
    rel.components().any(|c| {
        if let std::path::Component::Normal(s) = c {
            excluded.iter().any(|e| s == std::ffi::OsStr::new(e))
        } else {
            false
        }
    })
}

fn sha256_hex(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f =
        fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut f, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn file_mode(md: &fs::Metadata) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        return Some(md.mode());
    }
    #[cfg(not(unix))]
    {
        let _ = md;
        None
    }
}

fn new_checkpoint_id() -> String {
    let ts = unix_ts() as u64;
    let base36 = to_base36(ts);
    let suffix = uuid::Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .take(6)
        .collect::<String>();
    format!("{}-{}", base36, suffix)
}

fn to_base36(mut n: u64) -> String {
    const DIG: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push(DIG[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8_lossy(&out).to_string()
}

fn project_id(root: &Path) -> String {
    let canon = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let s = canon.to_string_lossy().to_string();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn unix_ts() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn git_available() -> bool {
        std::process::Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn checkpoint_snapshot_roundtrip() {
        let td = TempDir::new().unwrap();
        let root = td.path();
        fs::write(root.join("a.txt"), "1").unwrap();
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("sub").join("b.txt"), "2").unwrap();

        let store = SessionStore::new(root, "s1".into()).unwrap();
        store.init_file();

        let meta = create_checkpoint(root, &store, Some("snap".into())).unwrap();
        assert_eq!(meta.backend, CheckpointBackend::Snapshot);

        fs::write(root.join("a.txt"), "changed").unwrap();
        let _ = fs::remove_file(root.join("sub").join("b.txt"));
        fs::write(root.join("c.txt"), "3").unwrap();

        restore_checkpoint(root, &store, &meta).unwrap();
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "1");
        assert_eq!(
            fs::read_to_string(root.join("sub").join("b.txt")).unwrap(),
            "2"
        );
        assert!(!root.join("c.txt").exists());
    }

    #[test]
    fn checkpoint_git_roundtrip_tracked_untracked() {
        if !git_available() {
            eprintln!("skipping: git not available");
            return;
        }
        let td = TempDir::new().unwrap();
        let root = td.path();
        run_git_status_ok(root, &["init"]).unwrap();
        fs::write(root.join("t.txt"), "base").unwrap();
        run_git_status_ok(root, &["add", "t.txt"]).unwrap();
        run_git_status_ok(root, &["commit", "-m", "init"]).unwrap();

        let store = SessionStore::new(root, "s1".into()).unwrap();
        store.init_file();

        fs::write(root.join("t.txt"), "mod").unwrap();
        fs::write(root.join("u.txt"), "untracked").unwrap();

        let meta = create_checkpoint(root, &store, Some("git".into())).unwrap();
        assert_eq!(meta.backend, CheckpointBackend::Git);

        fs::write(root.join("t.txt"), "mod2").unwrap();
        fs::write(root.join("u2.txt"), "new").unwrap();

        restore_checkpoint(root, &store, &meta).unwrap();
        assert_eq!(fs::read_to_string(root.join("t.txt")).unwrap(), "mod");
        assert_eq!(fs::read_to_string(root.join("u.txt")).unwrap(), "untracked");
        assert!(!root.join("u2.txt").exists());
    }

    #[test]
    fn checkpoint_git_roundtrip_staged_and_unstaged() {
        if !git_available() {
            eprintln!("skipping: git not available");
            return;
        }
        let td = TempDir::new().unwrap();
        let root = td.path();
        run_git_status_ok(root, &["init"]).unwrap();
        fs::write(root.join("f.txt"), "0").unwrap();
        run_git_status_ok(root, &["add", "f.txt"]).unwrap();
        run_git_status_ok(root, &["commit", "-m", "init"]).unwrap();

        let store = SessionStore::new(root, "s1".into()).unwrap();
        store.init_file();

        // staged change
        fs::write(root.join("f.txt"), "1").unwrap();
        run_git_status_ok(root, &["add", "f.txt"]).unwrap();
        // unstaged change on top
        fs::write(root.join("f.txt"), "2").unwrap();

        let meta = create_checkpoint(root, &store, Some("mix".into())).unwrap();

        // mutate further
        fs::write(root.join("f.txt"), "3").unwrap();

        restore_checkpoint(root, &store, &meta).unwrap();

        // After restore, working tree should have unstaged content ("2"), and index should have staged content ("1").
        let wt = fs::read_to_string(root.join("f.txt")).unwrap();
        assert_eq!(wt, "2");
        let idx = run_git(root, &["show", ":f.txt"]).unwrap();
        assert_eq!(idx, "1");
    }
}
