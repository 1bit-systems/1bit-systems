//! SQLite-backed session storage for conversation history and thread management.
//!
//! Stores each conversation thread as a sequence of turns (user message + model
//! response), persisted to `~/.1bit/sessions/sessions.db`.  Supports listing,
//! creating, resuming, and archiving sessions — matching codewhale's session API.

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;
use uuid::Uuid;

/// A single session (conversation thread) with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
    pub turn_count: u32,
    pub archived: bool,
}

/// A single turn in a session (user message + assistant response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub tool_calls: Option<String>,
    pub created_at: String,
}

/// Session storage manager.
pub struct SessionStore {
    conn: Connection,
}

impl SessionStore {
    /// Open (or create) the session database.
    pub fn open() -> Result<Self> {
        let path = Self::db_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create sessions directory")?;
        }

        let conn = Connection::open(&path)
            .with_context(|| format!("Failed to open session database at {}", path.display()))?;

        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Path to the SQLite database.
    fn db_path() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Cannot find home directory")?;
        Ok(PathBuf::from(home).join(".1bit").join("sessions").join("sessions.db"))
    }

    /// Create tables if they don't exist.
    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch("
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL DEFAULT 'untitled',
                model       TEXT NOT NULL DEFAULT 'default',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                turn_count  INTEGER NOT NULL DEFAULT 0,
                archived    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS turns (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL DEFAULT '',
                tool_calls  TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_id, created_at);

            CREATE TABLE IF NOT EXISTS session_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        ")?;
        Ok(())
    }

    /// Create a new session.
    pub fn create_session(&self, name: &str, model: &str) -> Result<Session> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO sessions (id, name, model, created_at, updated_at, turn_count, archived)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, 0)",
            params![id, name, model, now, now],
        )?;

        Ok(Session {
            id,
            name: name.to_string(),
            model: model.to_string(),
            created_at: now.clone(),
            updated_at: now,
            turn_count: 0,
            archived: false,
        })
    }

    /// List all sessions, newest first.
    pub fn list_sessions(&self, include_archived: bool) -> Result<Vec<Session>> {
        let query = if include_archived {
            "SELECT id, name, model, created_at, updated_at, turn_count, archived
             FROM sessions ORDER BY updated_at DESC"
        } else {
            "SELECT id, name, model, created_at, updated_at, turn_count, archived
             FROM sessions WHERE archived = 0 ORDER BY updated_at DESC"
        };

        let mut stmt = self.conn.prepare(query)?;
        let rows = stmt.query_map([], |row| {
            Ok(Session {
                id: row.get(0)?,
                name: row.get(1)?,
                model: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                turn_count: row.get(5)?,
                archived: row.get::<_, i32>(6)? != 0,
            })
        })?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    /// Get a single session by ID.
    pub fn get_session(&self, id: &str) -> Result<Option<Session>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, model, created_at, updated_at, turn_count, archived
             FROM sessions WHERE id = ?1"
        )?;

        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Session {
                id: row.get(0)?,
                name: row.get(1)?,
                model: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                turn_count: row.get(5)?,
                archived: row.get::<_, i32>(6)? != 0,
            })
        })?;

        match rows.next() {
            Some(Ok(session)) => Ok(Some(session)),
            _ => Ok(None),
        }
    }

    /// Update session timestamp and turn count.
    fn touch_session(&self, id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE sessions SET updated_at = ?1, turn_count = (
                SELECT COUNT(*) FROM turns WHERE session_id = ?2
             ) WHERE id = ?2",
            params![now, id],
        )?;
        Ok(())
    }

    /// Archive a session (soft-delete).
    pub fn archive_session(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET archived = 1 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Add a user message turn.
    pub fn add_user_turn(&self, session_id: &str, content: &str) -> Result<Turn> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO turns (id, session_id, role, content, created_at)
             VALUES (?1, ?2, 'user', ?3, ?4)",
            params![id, session_id, content, now],
        )?;

        self.touch_session(session_id)?;

        Ok(Turn {
            id,
            session_id: session_id.to_string(),
            role: "user".into(),
            content: content.to_string(),
            tool_calls: None,
            created_at: now,
        })
    }

    /// Add an assistant response turn (with optional tool calls).
    pub fn add_assistant_turn(
        &self,
        session_id: &str,
        content: &str,
        tool_calls: Option<&str>,
    ) -> Result<Turn> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO turns (id, session_id, role, content, tool_calls, created_at)
             VALUES (?1, ?2, 'assistant', ?3, ?4, ?5)",
            params![id, session_id, content, tool_calls, now],
        )?;

        self.touch_session(session_id)?;

        Ok(Turn {
            id,
            session_id: session_id.to_string(),
            role: "assistant".into(),
            content: content.to_string(),
            tool_calls: tool_calls.map(|s| s.to_string()),
            created_at: now,
        })
    }

    /// Add a tool result turn.
    pub fn add_tool_turn(
        &self,
        session_id: &str,
        tool_call_id: &str,
        content: &str,
    ) -> Result<Turn> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let tool_calls = serde_json::json!([{"tool_call_id": tool_call_id}]).to_string();

        self.conn.execute(
            "INSERT INTO turns (id, session_id, role, content, tool_calls, created_at)
             VALUES (?1, ?2, 'tool', ?3, ?4, ?5)",
            params![id, session_id, content, tool_calls, now],
        )?;

        self.touch_session(session_id)?;

        Ok(Turn {
            id,
            session_id: session_id.to_string(),
            role: "tool".into(),
            content: content.to_string(),
            tool_calls: Some(tool_calls),
            created_at: now,
        })
    }

    /// Get all turns for a session, ordered by creation time.
    pub fn get_turns(&self, session_id: &str) -> Result<Vec<Turn>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, role, content, tool_calls, created_at
             FROM turns WHERE session_id = ?1 ORDER BY created_at ASC"
        )?;

        let rows = stmt.query_map(params![session_id], |row| {
            Ok(Turn {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                tool_calls: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        let mut turns = Vec::new();
        for row in rows {
            turns.push(row?);
        }
        Ok(turns)
    }

    /// Delete a session and all its turns.
    pub fn delete_session(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM turns WHERE session_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Get or create the last active session.
    pub fn get_or_create_active(&self, model: &str) -> Result<Session> {
        // Try to find the most recent non-archived session
        let sessions = self.list_sessions(false)?;
        if let Some(session) = sessions.into_iter().next() {
            return Ok(session);
        }
        // Create a new one
        let now = Utc::now();
        let name = format!("Chat {}", now.format("%Y-%m-%d %H:%M"));
        self.create_session(&name, model)
    }

    /// Convert session turns into NPU message history.
    pub fn turns_to_messages(&self, session_id: &str) -> Result<Vec<crate::npu::Message>> {
        let turns = self.get_turns(session_id)?;
        let messages: Vec<crate::npu::Message> = turns
            .into_iter()
            .filter_map(|t| match t.role.as_str() {
                "user" => Some(crate::npu::Message::user(t.content)),
                "assistant" => {
                    if let Some(tc_json) = t.tool_calls {
                        // Check if this assistant turn has tool calls
                        if let Ok(tcs) = serde_json::from_str::<Vec<serde_json::Value>>(&tc_json) {
                            if !tcs.is_empty() && t.content.is_empty() {
                                // This is just tool calls, no text — skip for now
                                // (the tool loop handles this)
                                return None;
                            }
                        }
                    }
                    Some(crate::npu::Message::assistant(t.content))
                }
                "tool" => {
                    // Parse tool_call_id from the stored tool_calls field
                    let tool_call_id = t
                        .tool_calls
                        .as_ref()
                        .and_then(|s| {
                            serde_json::from_str::<Vec<serde_json::Value>>(s).ok()
                        })
                        .and_then(|v| v.first().cloned())
                        .and_then(|v| v.get("tool_call_id").cloned())
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Some(crate::npu::Message::tool_result(tool_call_id, t.content))
                }
                _ => None,
            })
            .collect();
        Ok(messages)
    }
}
