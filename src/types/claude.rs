use serde::{Deserialize, Serialize};

// --- Cache Control ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

// --- Citation ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    #[serde(rename = "type")]
    pub citation_type: String,
    pub cited_text: String,
    pub document_index: u32,
    pub document_title: String,
    pub start_char_index: u32,
    pub end_char_index: u32,
}

// --- Image Source ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

// --- Document Source ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

// --- Search Result ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

// --- Thinking Config ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ThinkingConfig {
    #[serde(rename = "enabled")]
    Enabled { budget_tokens: u32 },
    #[serde(rename = "disabled")]
    Disabled,
}

// --- Content Block ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: ClaudeContent,
    },
    #[serde(rename = "web_search_result")]
    WebSearchResult {
        search_query: String,
        search_results: Vec<SearchResult>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

// --- Content: string or array ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClaudeContent {
    Text(String),
    Blocks(Vec<ClaudeContentBlock>),
}

// --- Message ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: ClaudeContent,
}

// --- System Prompt ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ClaudeTextBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTextBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// --- Tool ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

// --- Tool Choice ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
    #[serde(rename = "none")]
    None,
}

// --- Metadata ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// --- Usage ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// --- Messages Request ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessagesRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

// --- Messages Response ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ClaudeContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

// --- Token Counting ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTokenCountingRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTokenCountingResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub input_tokens: u32,
}

// --- Models ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeModelsResponse {
    pub data: Vec<ClaudeModel>,
    pub first_id: Option<String>,
    pub has_more: bool,
    pub last_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeModel {
    pub id: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub created_at: String,
    pub display_name: String,
}
