use std::path::Path;

use tree_sitter::{Language as TSLanguage, Parser, Tree};

use crate::semantic_search::types::{ChunkMetadata, CodeChunk, Language, SymbolType};

/// AST-aware code chunker using tree-sitter
pub struct Chunker {
    max_chunk_size: usize,
}

impl Chunker {
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Chunk a source file into meaningful code segments
    pub fn chunk_file(
        &self,
        content: &str,
        file_path: &Path,
        language: Language,
    ) -> Vec<CodeChunk> {
        // Try AST-aware chunking first
        if let Some(ts_language) = get_tree_sitter_language(language) {
            if let Some(chunks) = self.chunk_with_ast(content, file_path, language, ts_language) {
                return chunks;
            }
        }

        // Fall back to line-based chunking
        self.chunk_by_lines(content, file_path, language)
    }

    /// Chunk using tree-sitter AST
    fn chunk_with_ast(
        &self,
        content: &str,
        file_path: &Path,
        language: Language,
        ts_language: TSLanguage,
    ) -> Option<Vec<CodeChunk>> {
        let mut parser = Parser::new();
        parser.set_language(&ts_language).ok()?;
        let tree = parser.parse(content, None)?;

        let mut chunks = Vec::new();
        let mut chunk_id = 0u64;

        self.extract_chunks_from_tree(
            &tree,
            content,
            file_path,
            language,
            &mut chunks,
            &mut chunk_id,
        );

        if chunks.is_empty() {
            return None;
        }

        Some(chunks)
    }

    /// Extract meaningful chunks from the AST
    fn extract_chunks_from_tree(
        &self,
        tree: &Tree,
        content: &str,
        file_path: &Path,
        language: Language,
        chunks: &mut Vec<CodeChunk>,
        chunk_id: &mut u64,
    ) {
        let root = tree.root_node();
        let mut cursor = root.walk();

        // Get all top-level definitions
        let content_bytes = content.as_bytes();
        for child in root.children(&mut cursor) {
            if let Some((symbol_type, symbol_name)) = classify_node(&child, language, content_bytes)
            {
                let start_byte = child.start_byte();
                let end_byte = child.end_byte();
                let node_content = &content[start_byte..end_byte];

                // If the chunk is too large, split it
                if node_content.len() > self.max_chunk_size {
                    self.split_large_chunk(
                        node_content,
                        file_path,
                        language,
                        child.start_position().row + 1,
                        Some(symbol_name.clone()),
                        Some(symbol_type),
                        chunks,
                        chunk_id,
                    );
                } else {
                    let start_line = child.start_position().row + 1;
                    let end_line = child.end_position().row + 1;

                    chunks.push(CodeChunk {
                        id: *chunk_id,
                        content: node_content.to_string(),
                        metadata: ChunkMetadata {
                            file_path: file_path.to_path_buf(),
                            start_line,
                            end_line,
                            language,
                            symbol_name: Some(symbol_name),
                            symbol_type: Some(symbol_type),
                        },
                    });
                    *chunk_id += 1;
                }
            }
        }

        // If no top-level definitions found, fall back to chunking the whole file
        if chunks.is_empty() {
            self.chunk_by_lines_into(content, file_path, language, chunks, chunk_id);
        }
    }

    /// Split a large chunk into smaller pieces
    fn split_large_chunk(
        &self,
        content: &str,
        file_path: &Path,
        language: Language,
        base_line: usize,
        symbol_name: Option<String>,
        symbol_type: Option<SymbolType>,
        chunks: &mut Vec<CodeChunk>,
        chunk_id: &mut u64,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        let mut current_chunk = String::new();
        let mut chunk_start_line = base_line;
        let mut current_line = base_line;

        for line in lines {
            if current_chunk.len() + line.len() + 1 > self.max_chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(CodeChunk {
                    id: *chunk_id,
                    content: current_chunk.clone(),
                    metadata: ChunkMetadata {
                        file_path: file_path.to_path_buf(),
                        start_line: chunk_start_line,
                        end_line: current_line - 1,
                        language,
                        symbol_name: symbol_name.clone(),
                        symbol_type,
                    },
                });
                *chunk_id += 1;
                current_chunk.clear();
                chunk_start_line = current_line;
            }

            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
            current_line += 1;
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(CodeChunk {
                id: *chunk_id,
                content: current_chunk,
                metadata: ChunkMetadata {
                    file_path: file_path.to_path_buf(),
                    start_line: chunk_start_line,
                    end_line: current_line - 1,
                    language,
                    symbol_name,
                    symbol_type,
                },
            });
            *chunk_id += 1;
        }
    }

    /// Fall back to line-based chunking
    fn chunk_by_lines(
        &self,
        content: &str,
        file_path: &Path,
        language: Language,
    ) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();
        let mut chunk_id = 0u64;
        self.chunk_by_lines_into(content, file_path, language, &mut chunks, &mut chunk_id);
        chunks
    }

    fn chunk_by_lines_into(
        &self,
        content: &str,
        file_path: &Path,
        language: Language,
        chunks: &mut Vec<CodeChunk>,
        chunk_id: &mut u64,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        let mut current_chunk = String::new();
        let mut chunk_start_line = 1;
        let mut current_line = 1;

        for line in lines {
            if current_chunk.len() + line.len() + 1 > self.max_chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(CodeChunk {
                    id: *chunk_id,
                    content: current_chunk.clone(),
                    metadata: ChunkMetadata {
                        file_path: file_path.to_path_buf(),
                        start_line: chunk_start_line,
                        end_line: current_line - 1,
                        language,
                        symbol_name: None,
                        symbol_type: None,
                    },
                });
                *chunk_id += 1;
                current_chunk.clear();
                chunk_start_line = current_line;
            }

            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
            current_line += 1;
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(CodeChunk {
                id: *chunk_id,
                content: current_chunk,
                metadata: ChunkMetadata {
                    file_path: file_path.to_path_buf(),
                    start_line: chunk_start_line,
                    end_line: current_line - 1,
                    language,
                    symbol_name: None,
                    symbol_type: None,
                },
            });
            *chunk_id += 1;
        }
    }
}

/// Get the tree-sitter language for a given Language enum
fn get_tree_sitter_language(language: Language) -> Option<TSLanguage> {
    match language {
        Language::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
        Language::Python => Some(tree_sitter_python::LANGUAGE.into()),
        Language::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
        Language::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        Language::Go => Some(tree_sitter_go::LANGUAGE.into()),
        Language::C => Some(tree_sitter_c::LANGUAGE.into()),
        Language::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
        Language::Java => Some(tree_sitter_java::LANGUAGE.into()),
        Language::Ruby => Some(tree_sitter_ruby::LANGUAGE.into()),
        Language::Unknown => None,
    }
}

/// Classify a tree-sitter node and extract its name
fn classify_node(
    node: &tree_sitter::Node,
    language: Language,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    let kind = node.kind();

    match language {
        Language::Rust => classify_rust_node(node, kind, content),
        Language::Python => classify_python_node(node, kind, content),
        Language::JavaScript | Language::TypeScript => classify_js_node(node, kind, content),
        Language::Go => classify_go_node(node, kind, content),
        Language::C | Language::Cpp => classify_c_cpp_node(node, kind, content),
        Language::Java => classify_java_node(node, kind, content),
        Language::Ruby => classify_ruby_node(node, kind, content),
        Language::Unknown => None,
    }
}

fn classify_rust_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "function_item" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Function, name))
        }
        "struct_item" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Struct, name))
        }
        "enum_item" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Enum, name))
        }
        "trait_item" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Trait, name))
        }
        "impl_item" => {
            // For impl blocks, try to get the type name
            let name =
                find_child_by_field(node, "type", content).or_else(|| Some("impl".to_string()))?;
            Some((SymbolType::Impl, name))
        }
        "mod_item" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Module, name))
        }
        _ => None,
    }
}

fn classify_python_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "function_definition" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Function, name))
        }
        "class_definition" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Class, name))
        }
        _ => None,
    }
}

fn classify_js_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "function_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Function, name))
        }
        "class_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Class, name))
        }
        "method_definition" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Method, name))
        }
        "arrow_function" | "function_expression" => {
            // These are usually assigned to variables
            Some((SymbolType::Function, "anonymous".to_string()))
        }
        "interface_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Interface, name))
        }
        "type_alias_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Other, name))
        }
        _ => None,
    }
}

fn classify_go_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "function_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Function, name))
        }
        "method_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Method, name))
        }
        "type_declaration" => {
            // Could be struct, interface, or type alias
            Some((SymbolType::Struct, "type".to_string()))
        }
        _ => None,
    }
}

fn classify_c_cpp_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "function_definition" => {
            let name = find_child_by_field(node, "declarator", content)
                .or_else(|| find_function_name(node, content))?;
            Some((SymbolType::Function, name))
        }
        "struct_specifier" => {
            let name = find_child_by_field(node, "name", content)
                .unwrap_or_else(|| "anonymous_struct".to_string());
            Some((SymbolType::Struct, name))
        }
        "class_specifier" => {
            let name = find_child_by_field(node, "name", content)
                .unwrap_or_else(|| "anonymous_class".to_string());
            Some((SymbolType::Class, name))
        }
        "enum_specifier" => {
            let name = find_child_by_field(node, "name", content)
                .unwrap_or_else(|| "anonymous_enum".to_string());
            Some((SymbolType::Enum, name))
        }
        _ => None,
    }
}

fn classify_java_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "method_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Method, name))
        }
        "class_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Class, name))
        }
        "interface_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Interface, name))
        }
        "enum_declaration" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Enum, name))
        }
        _ => None,
    }
}

fn classify_ruby_node(
    node: &tree_sitter::Node,
    kind: &str,
    content: &[u8],
) -> Option<(SymbolType, String)> {
    match kind {
        "method" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Method, name))
        }
        "class" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Class, name))
        }
        "module" => {
            let name = find_child_by_field(node, "name", content)?;
            Some((SymbolType::Module, name))
        }
        _ => None,
    }
}

/// Helper to find a named child field and extract its text
fn find_child_by_field(node: &tree_sitter::Node, field: &str, content: &[u8]) -> Option<String> {
    node.child_by_field_name(field)
        .map(|n| n.utf8_text(content).ok())
        .flatten()
        .map(|s| s.to_string())
}

/// Helper to extract function name from complex declarators (C/C++)
fn find_function_name(node: &tree_sitter::Node, content: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "function_declarator" {
            return find_child_by_field(&child, "declarator", content);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("txt"), Language::Unknown);
    }

    #[test]
    fn test_chunker_rust() {
        let chunker = Chunker::new(2000);
        let content = r#"
fn hello() {
    println!("Hello");
}

struct Foo {
    bar: i32,
}

impl Foo {
    fn new() -> Self {
        Foo { bar: 0 }
    }
}
"#;
        let chunks = chunker.chunk_file(content, Path::new("test.rs"), Language::Rust);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunker_python() {
        let chunker = Chunker::new(2000);
        let content = r#"
def hello():
    print("Hello")

class Foo:
    def __init__(self):
        self.bar = 0
"#;
        let chunks = chunker.chunk_file(content, Path::new("test.py"), Language::Python);
        assert!(!chunks.is_empty());
    }
}
