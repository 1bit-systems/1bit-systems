//! Order management — persistence, email notifications.
//!
//! Mirrors the Python version's order handling:
//! - Pending orders persisted to disk (survives daemon restart)
//! - Fulfilled orders logged to `orders.json`
//! - Email notification via SMTP or local sendmail

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

// ── Data types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderItem {
    pub product: String,
    #[serde(default)]
    pub size: String,
    #[serde(default = "default_qty")]
    pub qty: u32,
    #[serde(default)]
    pub price: u64, // cents
}

fn default_qty() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub items: Vec<OrderItem>,
    pub total: u64, // cents
    #[serde(default)]
    pub customer_name: String,
    #[serde(default)]
    pub customer_email: String,
    #[serde(default)]
    pub shipping_address: String,
    #[serde(default)]
    pub stripe_session_id: String,
    #[serde(default)]
    pub logged_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckoutRequest {
    pub items: Vec<CheckoutItem>,
    #[serde(default = "default_success_url")]
    pub success_url: String,
    #[serde(default = "default_cancel_url")]
    pub cancel_url: String,
}

fn default_success_url() -> String {
    "https://1bit.systems/store/success".to_string()
}
fn default_cancel_url() -> String {
    "https://1bit.systems/store".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckoutItem {
    pub price_data: Option<PriceData>,
    pub quantity: Option<u32>,
    #[serde(default)]
    pub size: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub currency: Option<String>,
    pub product_data: Option<ProductData>,
    pub unit_amount: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductData {
    pub name: Option<String>,
}

// ── Order Manager ───────────────────────────────────────────────────

pub struct OrderManager {
    pending_path: PathBuf,
    log_path: PathBuf,
    /// Pending orders keyed by Stripe session ID.
    pending: Arc<RwLock<HashMap<String, Order>>>,
    // SMTP config (optional)
    smtp_host: String,
    _smtp_port: u16,
    smtp_user: String,
    smtp_pass: String,
    notify_email: String,
    notify_cc: String,
}

impl OrderManager {
    pub fn new(
        repo_root: &PathBuf,
        smtp_host: String,
        smtp_port: u16,
        smtp_user: String,
        smtp_pass: String,
        notify_email: String,
        notify_cc: String,
    ) -> Self {
        Self {
            pending_path: repo_root.join(".pending-orders.json"),
            log_path: repo_root.join("orders.json"),
            pending: Arc::new(RwLock::new(HashMap::new())),
            smtp_host,
            _smtp_port: smtp_port,
            smtp_user,
            smtp_pass,
            notify_email,
            notify_cc,
        }
    }

    /// Load pending orders from disk (survives daemon restart).
    pub async fn load_pending(&self) -> Result<()> {
        if !self.pending_path.exists() {
            return Ok(());
        }
        let data = tokio::fs::read_to_string(&self.pending_path).await
            .context("Failed to read pending orders")?;
        let orders: HashMap<String, Order> = serde_json::from_str(&data)
            .context("Failed to parse pending orders")?;
        *self.pending.write().await = orders;
        info!("Loaded {} pending orders from disk", self.pending.read().await.len());
        Ok(())
    }

    /// Save pending orders to disk.
    pub async fn save_pending(&self) -> Result<()> {
        let orders = self.pending.read().await;
        let data = serde_json::to_string_pretty(&*orders)?;
        tokio::fs::write(&self.pending_path, data).await
            .context("Failed to write pending orders")?;
        // Ensure orders.json exists
        if !self.log_path.exists() {
            tokio::fs::write(&self.log_path, "[]").await
                .context("Failed to create orders.json")?;
        }
        Ok(())
    }

    /// Add a pending order.
    pub async fn add_pending(&self, session_id: &str, order: Order) -> Result<()> {
        self.pending.write().await.insert(session_id.to_string(), order);
        self.save_pending().await
    }

    /// Remove and return a pending order (on fulfillment).
    pub async fn take_pending(&self, session_id: &str) -> Option<Order> {
        let order = self.pending.write().await.remove(session_id);
        if order.is_some() {
            let _ = self.save_pending().await;
        }
        order
    }

    /// Log a fulfilled order to orders.json.
    pub async fn log_order(&self, order: &Order) -> Result<()> {
        let mut orders: Vec<Order> = if self.log_path.exists() {
            let data = tokio::fs::read_to_string(&self.log_path).await
                .context("Failed to read orders log")?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut order = order.clone();
        order.logged_at = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
        orders.push(order);

        let data = serde_json::to_string_pretty(&orders)?;
        if let Some(parent) = self.log_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create orders log directory")?;
        }
        tokio::fs::write(&self.log_path, data).await
            .context("Failed to write orders log")?;
        Ok(())
    }

    /// Send order notification email.
    pub async fn send_order_email(&self, order: &Order) -> Result<()> {
        let subject = format!(
            "🛒 1bit Store Order — {}",
            sanitize_header(&order.customer_name)
        );
        let body = format!(
            "═══════════════════════════════════════\r\n\
             \x20 NEW ORDER — 1bit.systems Store\r\n\
             ═══════════════════════════════════════\r\n\
             \r\n\
             {items}\r\n\
             \x20 Total:     ${total:.2}\r\n\
             \x20 Customer:  {name}\r\n\
             \x20 Email:     {email}\r\n\
             \x20 Shipping:  {ship}\r\n\
             \x20 Stripe ID: {stripe}\r\n\
             \r\n\
             \x20 Free extras (included): stickers + lanyard + thank you card + mystery sticker\r\n\
             \r\n\
             ═══════════════════════════════════════\r\n\
             \x20 Ship it. —bong-water-water-bong\r\n\
             ═══════════════════════════════════════\r\n",
            items = format_order_items(&order.items),
            total = order.total as f64 / 100.0,
            name = sanitize_header(&order.customer_name),
            email = sanitize_header(&order.customer_email),
            ship = sanitize_header(&order.shipping_address),
            stripe = &order.stripe_session_id,
        );

        let to = if !self.notify_cc.is_empty() {
            vec![self.notify_email.clone(), self.notify_cc.clone()]
        } else {
            vec![self.notify_email.clone()]
        };

        // Try SMTP first, fall back to sendmail
        if !self.smtp_host.is_empty() && self.smtp_host != "localhost" {
            self.send_via_smtp(&to, &subject, &body).await
        } else {
            self.send_via_sendmail(&to, &subject, &body).await
        }
    }

    async fn send_via_smtp(&self, to: &[String], subject: &str, body: &str) -> Result<()> {
        use lettre::{
            transport::smtp::authentication::Credentials,
            AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor,
        };

        let from = "store@1bit.systems".to_string();
        let msg = Message::builder()
            .from(from.parse().context("Invalid from address")?)
            .to(to.first().unwrap().parse().context("Invalid to address")?)
            .subject(subject)
            .body(body.to_string())
            .context("Failed to build email message")?;

        let mailer = if !self.smtp_user.is_empty() {
            let creds = Credentials::new(self.smtp_user.clone(), self.smtp_pass.clone());
            AsyncSmtpTransport::<Tokio1Executor>::relay(&self.smtp_host)
                .context("Failed to configure SMTP relay")?
                .credentials(creds)
                .build()
        } else {
            AsyncSmtpTransport::<Tokio1Executor>::relay(&self.smtp_host)
                .context("Failed to configure SMTP relay")?
                .build()
        };

        mailer.send(msg).await
            .context("Failed to send email via SMTP")?;

        info!("Order email sent to {}", to.first().unwrap());
        Ok(())
    }

    async fn send_via_sendmail(&self, _to: &[String], _subject: &str, body: &str) -> Result<()> {
        // Use local sendmail
        let mut child = tokio::process::Command::new("sendmail")
            .arg("-t")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn sendmail")?;

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(body.as_bytes()).await
                .context("Failed to write to sendmail stdin")?;
            // Drop stdin to close it
            drop(stdin);
        }

        let status = child.wait().await
            .context("Failed to wait for sendmail")?;
        if !status.success() {
            warn!("sendmail exited with status: {status}");
        }
        info!("Order email sent via sendmail");
        Ok(())
    }

    /// Fulfill an order: log it and send notification.
    pub async fn fulfill_order(&self, order: &Order) -> Result<()> {
        self.log_order(order).await?;

        info!(
            "\n{sep}\n  🛒 ORDER PAID — ${total:.2}\n{items}  Ship to: {name} — {addr}\n{sep}",
            sep = "=".repeat(60),
            total = order.total as f64 / 100.0,
            items = format_order_items(&order.items)
                .lines()
                .map(|l| format!("    {l}\n"))
                .collect::<String>(),
            name = order.customer_name,
            addr = order.shipping_address,
        );

        // Try sending email, but don't fail if it doesn't work
        if let Err(e) = self.send_order_email(order).await {
            warn!("Order email failed: {e} (order saved to orders.json)");
        }

        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn sanitize_header(val: &str) -> String {
    val.replace('\r', "").replace('\n', " ").trim().to_string()
}

fn format_order_items(items: &[OrderItem]) -> String {
    let mut lines = Vec::new();
    for item in items {
        let size = if item.size.is_empty() {
            String::new()
        } else {
            format!(" ({})", item.size)
        };
        lines.push(format!(
            "  {}× {}{}  —  ${:.2} each",
            item.qty,
            item.product,
            size,
            item.price as f64 / 100.0
        ));
    }
    lines.join("\r\n")
}
