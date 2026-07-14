#!/usr/bin/env python3
"""
npu-gpu-cpud — Unified Control Plane Daemon (v3 with C++ engine)

Routes inference requests to NPU, GPU, or CPU based on policy.
Backends:
  NPU: Open-source C++ engine (MIT, 97 tok/s Qwen3-0.6B)
  GPU: ROCm/WMMA GPU engine
  CPU: Lemonade CPU (lemond --backend cpu)

Policy (model_size → device):
  < 2B params  → NPU (lowest power)
  2B-8B params → GPU (fastest compute)
  > 8B params  → CPU (fallback)

Usage:
  sudo ./daemon/npu-gpu-cpud.py [--port PORT] [--npu-port PORT] [--gpu-port PORT]

Part of 1bit-systems. Lives in daemon/ in the repo root.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import urllib.request
import urllib.error
from tokenizers import Tokenizer

# Paths
NPU_ENGINE_BIN = os.environ.get("NPU_ENGINE_BIN", "/home/bcloud/engine/npu/build/npu_engine_server")
NPU_MODEL_PATH = os.environ.get("NPU_MODEL_PATH", "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx")
NPU_TOKENIZER_PATH = os.environ.get("NPU_TOKENIZER_PATH", "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")

# Stripe integration — uses raw HTTPS (no SDK needed)
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")

# Order notification emails
ORDER_NOTIFY_EMAIL = os.environ.get("ORDER_NOTIFY_EMAIL", "sales@1bit.systems")  # supplier
ORDER_NOTIFY_CC = os.environ.get("ORDER_NOTIFY_CC", "")                          # you
SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "25"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
ORDER_LOG = os.path.join(os.path.dirname(__file__), "..", "orders.json")

def _send_order_email(order: dict):
    """Send order notification via SMTP or local sendmail."""
    subject = f"🛒 1bit Store Order — {order.get('customer_name','Customer')}"
    lines = [f"From: 1bit Store <store@1bit.systems>",
             f"To: {ORDER_NOTIFY_EMAIL}"]
    if ORDER_NOTIFY_CC:
        lines.append(f"Cc: {ORDER_NOTIFY_CC}")
    lines += [f"Subject: {subject}", "Content-Type: text/plain; charset=utf-8", ""]
    lines.append("═══════════════════════════════════════")
    lines.append("  NEW ORDER — 1bit.systems Store")
    lines.append("═══════════════════════════════════════")
    lines.append("")
    for item in order.get("items", []):
        name = item.get("product", "Merch")
        size = f" ({item.get('size')})" if item.get("size") else ""
        qty = item.get("qty", 1)
        price = item.get("price", 0)
        lines.append(f"  {qty}× {name}{size}  —  ${price/100:.2f} each")
    lines.append("")
    lines.append(f"  Total:     ${order.get('total',0)/100:.2f}")
    lines.append(f"  Customer:  {order.get('customer_name','?')}")
    lines.append(f"  Email:     {order.get('customer_email','?')}")
    lines.append(f"  Shipping:  {order.get('shipping_address','?')}")
    lines.append(f"  Stripe ID: {order.get('stripe_session_id','?')}")
    lines.append("")
    lines.append("  Free extras (included): stickers + lanyard + thank you card + mystery sticker")
    lines.append("")
    lines.append("═══════════════════════════════════════")
    lines.append("  Ship it. —bong-water-water-bong")
    lines.append("═══════════════════════════════════════")
    msg = "\r\n".join(lines)
    try:
        import smtplib
        if SMTP_HOST == "localhost":
            # Try local sendmail
            import subprocess as sp
            sp.run(["sendmail", "-t"], input=msg.encode(), timeout=10)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
                s.starttls()
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASS)
                s.sendmail("store@1bit.systems",
                           [e for e in [ORDER_NOTIFY_EMAIL, ORDER_NOTIFY_CC] if e],
                           msg.encode())
        print(f"  ✉️  Order email sent to {ORDER_NOTIFY_EMAIL}", flush=True)
    except Exception as e:
        print(f"  ⚠️  Email failed: {e} (order saved to {ORDER_LOG})", flush=True)

def _log_order(order: dict):
    """Persist order to JSON log."""
    orders = []
    try:
        with open(ORDER_LOG) as f:
            orders = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    order["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    orders.append(order)
    os.makedirs(os.path.dirname(ORDER_LOG), exist_ok=True)
    with open(ORDER_LOG, "w") as f:
        json.dump(orders, f, indent=2)

# Pending orders: session_id → order details (persisted to disk)
_pending_orders: dict[str, dict] = {}
PENDING_LOG = os.path.join(os.path.dirname(__file__), "..", ".pending-orders.json")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

def _load_pending():
    """Restore pending orders from disk (survives daemon restart)."""
    global _pending_orders
    try:
        with open(PENDING_LOG) as f:
            _pending_orders = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _pending_orders = {}

def _save_pending():
    """Persist pending orders to disk."""
    with open(PENDING_LOG, "w") as f:
        json.dump(_pending_orders, f)
    # Also create orders.json if it doesn't exist
    if not os.path.exists(ORDER_LOG):
        with open(ORDER_LOG, "w") as f:
            json.dump([], f)

# Load any pending orders from previous daemon runs
_load_pending()

def _fulfill_order(order: dict):
    """Send notification and log order when payment completes."""
    _log_order(order)

    # Send notification email
    print(f"\n{'='*60}", flush=True)
    print(f"  🛒 ORDER PAID — {order.get('total',0)/100:.2f}", flush=True)
    for item in order.get("items", []):
        print(f"    {item.get('qty',1)}× {item.get('product','?')} {item.get('size','')}", flush=True)
    print(f"  Ship to: {order.get('customer_name','?')} — {order.get('shipping_address','?')}", flush=True)
    print(f"{'='*60}\n", flush=True)

    _send_order_email(order)

def _handle_stripe_webhook(body: bytes, sig_header: str) -> dict:
    """Process Stripe webhook event. Returns {ok: true} or {error: ...}."""
    import hmac, hashlib
    event = json.loads(body)
    event_type = event.get("type", "")

    # Verify signature if secret is configured
    if STRIPE_WEBHOOK_SECRET and sig_header:
        try:
            # Stripe signature format: t=<timestamp>,v1=<sig>,v0=<sig>,...
            parts = dict(p.split("=", 1) for p in sig_header.split(","))
            t = parts["t"]
            s = parts["v1"]
            signed = f"{t}.{body.decode()}"
            expected = hmac.new(
                STRIPE_WEBHOOK_SECRET.encode(), signed.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, s):
                return {"error": "Invalid signature", "status": 403}
        except Exception:
            return {"error": "Bad signature header", "status": 403}

    if event_type == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        sid = session.get("id", "")
        customer = session.get("customer_details", {})
        shipping = session.get("shipping_details", {})
        addr = shipping.get("address", {})

        # Build order from webhook data
        order = _pending_orders.pop(sid, None)
        if not order:
            # Reconstruct order from line items if we missed the creation
            line_items = session.get("display_items", [])
            order = {
                "items": [{"product": li.get("custom",{}).get("name","Merch"),
                           "qty": li.get("quantity",1),
                           "price": li.get("amount_total",0)} for li in line_items],
                "total": session.get("amount_total", 0),
                "customer_name": customer.get("name", shipping.get("name", "?")),
                "customer_email": customer.get("email", "?"),
                "shipping_address": f"{shipping.get('name','')} {addr.get('line1','')} {addr.get('city','')} {addr.get('state','')} {addr.get('country','')}",
            }
        else:
            _save_pending()  # persist removal from pending
            # Enrich with real customer details from Stripe
            order["customer_name"] = customer.get("name", shipping.get("name", order.get("customer_name", "?")))
            order["customer_email"] = customer.get("email", "?")
            order["shipping_address"] = f"{shipping.get('name','')} {addr.get('line1','')} {addr.get('city','')} {addr.get('state','')} {addr.get('country','')}".strip() or order.get("shipping_address", "?")

        order["stripe_session_id"] = sid
        _fulfill_order(order)
        return {"ok": True}

    return {"ok": True, "skipped": event_type}

def _stripe_create_checkout(items: list, success_url: str, cancel_url: str) -> dict:
    """Create a Stripe Checkout Session via REST API."""
    import ssl, http.client
    body = urllib.parse.urlencode({
        "mode": "payment",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "payment_method_types[]": "card",
    })
    for i, item in enumerate(items):
        prefix = f"line_items[{i}]"
        body += "&" + urllib.parse.urlencode({
            f"{prefix}[price_data][currency]": item.get("price_data", {}).get("currency", "usd"),
            f"{prefix}[price_data][product_data][name]": item.get("price_data", {}).get("product_data", {}).get("name", "Merch"),
            f"{prefix}[price_data][unit_amount]": str(item.get("price_data", {}).get("unit_amount", 0)),
            f"{prefix}[quantity]": str(item.get("quantity", 1)),
        })
    for j, country in enumerate([
        "US","CA","GB","DE","FR","AU","JP","BR","MX","NL","SE",
        "NO","DK","FI","CH","AT","BE","IE","PT","ES","IT","PL",
        "CZ","RO","GR","NZ","SG","HK","KR","IN"
    ]):
        body += "&" + urllib.parse.urlencode({f"shipping_address_collection[allowed_countries][{j}]": country})

    conn = http.client.HTTPSConnection("api.stripe.com", timeout=15, context=ssl.create_default_context())
    auth = f"Bearer {STRIPE_SECRET_KEY}"
    req = conn.request(
        "POST", "/v1/checkout/sessions",
        body=body.encode(),
        headers={
            "Authorization": auth,
            "Content-Type": "application/x-www-form-urlencoded",
        }
    )
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    if resp.status >= 400:
        return {"error": data.get("error", {}).get("message", f"HTTP {resp.status}")}
    return {"url": data.get("url"), "id": data.get("id")}

def _reject_json_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")

def _loads_json(body: bytes):
    return json.loads(body, parse_constant=_reject_json_constant)

def _parse_token(value) -> int:
    if isinstance(value, bool):
        raise ValueError("token must be an integer")
    if isinstance(value, int):
        token = value
    elif isinstance(value, str):
        if not value.strip().isdigit():
            raise ValueError(f"invalid token: {value!r}")
        token = int(value)
    else:
        raise ValueError("token must be an integer")
    if token < 0:
        raise ValueError("token must be non-negative")
    return token

def _parse_token_list(values) -> list[int]:
    if not isinstance(values, list):
        raise ValueError("tokens must be a list")
    return [_parse_token(value) for value in values]

# ---------------------------------------------------------------------------
# Backend lifecycle
# ---------------------------------------------------------------------------

class NPUBackend:
    """NPU inference via open-source C++ engine (MIT, 97 tok/s).

    Starts npu_engine_server as a subprocess, communicates via stdin/stdout
    JSON protocol. Tokenizes chat messages with HuggingFace tokenizers.
    """

    ENGINE_BIN = NPU_ENGINE_BIN
    MODEL_PATH = NPU_MODEL_PATH
    TOKENIZER_PATH = NPU_TOKENIZER_PATH

    def __init__(self, port: int = 0):
        # port is unused — we talk to engine via stdin/stdout, not HTTP
        self.port = port or 52625
        self.process: Optional[subprocess.Popen] = None
        self.tokenizer: Optional[Tokenizer] = None
        self._lock = threading.Lock()

    def start(self):
        if not os.path.exists(self.ENGINE_BIN):
            print(f"  ⚠️  NPU engine binary not found: {self.ENGINE_BIN}")
            print(f"     Run: cd /home/bcloud/engine/npu && make")
            return
        if not os.path.exists(self.TOKENIZER_PATH):
            print(f"  ⚠️  Tokenizer not found: {self.TOKENIZER_PATH}")
            return

        print(f"  Starting NPU C++ engine (97 tok/s, MIT)...")

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(self.TOKENIZER_PATH)
        self.tokenizer.no_truncation()
        print(f"    Tokenizer loaded ({self.TOKENIZER_PATH})")

        # Start engine subprocess
        self.process = subprocess.Popen(
            [self.ENGINE_BIN],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Wait for engine to be ready (reads model + xclbins, ~8s)
        # Engine writes status to stderr; wait for "Ready." line
        t0 = time.time()
        ready = False
        while time.time() - t0 < 30:
            line = self.process.stderr.readline()
            if not line:
                break
            print(f"    {line.rstrip()}", flush=True)
            if "Ready." in line:
                ready = True
                break

        if ready:
            print(f"  NPU C++ engine ready (pid={self.process.pid}, {time.time()-t0:.0f}s)")
        else:
            print(f"  ⚠️  NPU engine failed to start within 30s")
            self.stop()

    def stop(self):
        with self._lock:
            if self.process:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
                self.process = None
            self.tokenizer = None

    def _build_prompt(self, messages: list) -> str:
        """Build Qwen ChatML prompt from messages."""
        parts = []
        system_set = False
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                system_set = True
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if not system_set:
            parts.insert(0, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        """Run inference via C++ engine, return OpenAI-compatible response."""
        if not self.process or not self.tokenizer:
            return {"error": "NPU engine not running", "x-device": "npu"}

        with self._lock:
            try:
                # Build prompt and tokenize
                prompt = self._build_prompt(messages)
                encoded = self.tokenizer.encode(prompt)
                tokens = encoded.ids

                max_new = kwargs.get("max_tokens", 256)

                if len(tokens) > 4096:
                    return {"error": "Prompt too long", "x-device": "npu"}

                # Send to engine
                request = json.dumps({"tokens": tokens, "max_new_tokens": max_new})
                self.process.stdin.write(request + "\n")
                self.process.stdin.flush()

                # Read response
                line = self.process.stdout.readline()
                if not line:
                    return {"error": "Engine returned empty response", "x-device": "npu"}

                resp = json.loads(line)
                out_tokens = resp.get("tokens", [])

                # Decode tokens to text, stopping at EOS
                generated = ""
                eos_id = 151645  # <|im_end|>
                filtered = []
                for t in out_tokens:
                    if t == eos_id:
                        break
                    filtered.append(t)
                if filtered:
                    generated = self.tokenizer.decode(filtered, skip_special_tokens=True)

                prompt_tokens = len(tokens)
                completion_tokens = len(filtered)

                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": generated},
                        "finish_reason": "stop" if (out_tokens and out_tokens[-1] == eos_id) else "length",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "x-device": "npu",
                    "model": model,
                }

            except Exception as e:
                return {"error": str(e), "x-device": "npu"}

class GPUBackend:
    """Lemonade server on GPU (port 13305 by default)"""
    def __init__(self, port: int = 13305):
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        print(f"  Starting GPU backend (lemond) on port {self.port}...")
        cache_dir = f"/tmp/lemonade-gpu-{self.port}"
        self.process = subprocess.Popen(
            ["lemond", cache_dir, "--port", str(self.port), "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Wait for server to be ready
        for _ in range(30):
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/v1/health", timeout=2)
                if r.status == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        print(f"  GPU backend running (pid={self.process.pid})")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        body = {"model": model, "messages": messages, **kwargs}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=600)
        return json.loads(resp.read())

# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def estimate_model_size(model_name: str) -> float:
    """Estimate model size in billions of parameters from name."""
    model_lower = model_name.lower()
    for pattern, size in [
        ("0.5b", 0.5), ("0.6b", 0.6), ("0.8b", 0.8),
        ("1b", 1), ("1.2b", 1.2), ("1.5b", 1.5), ("1.7b", 1.7),
        ("2b", 2), ("2.6b", 2.6), ("3b", 3), ("4b", 4),
        ("7b", 7), ("8b", 8), ("9b", 9),
        ("13b", 13), ("14b", 14),
        ("20b", 20), ("34b", 34), ("70b", 70),
    ]:
        if pattern in model_lower:
            return size
    return 7  # default

def select_device(model_size_b: float) -> str:
    """Returns 'npu', 'gpu', or 'cpu' based on model size."""
    if model_size_b < 2:
        return "npu"
    elif model_size_b <= 8:
        return "gpu"
    else:
        return "cpu"

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

backends: dict = {}
start_time = time.time()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/store" or self.path == "/store/":
            self._serve_store()
            return
        if self.path == "/v1/health":
            self._json(200, {
                "status": "ok",
                "uptime_sec": int(time.time() - start_time),
                "devices": {
                    "npu": {"backend": "C++ engine (97 tok/s, MIT)", "port": backends["npu"].port,
                            "available": backends["npu"].process is not None},
                    "gpu": {"backend": "ROCm/WMMA GPU engine", "port": backends["gpu"].port,
                            "available": backends["gpu"].process is not None},
                    "cpu": {"backend": "Lemonade (CPU)", "available": True},
                },
                "policy": {
                    "< 2B params": "npu",
                    "2B-8B params": "gpu",
                    "> 8B params": "cpu",
                },
            })
        elif self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [
                    {"id": "Qwen3-0.6B-NPU2", "object": "model", "owned_by": "npu"},
                    {"id": "Qwen3-8B-NPU2", "object": "model", "owned_by": "npu"},
                    {"id": "Llama-3.1-8B-NPU2", "object": "model", "owned_by": "npu"},
                    {"id": "Gemma4-E2B-IT-NPU2", "object": "model", "owned_by": "npu"},
                ]
            })
        else:
            self._json(404, {"error": "Not found"})

    def do_OPTIONS(self):
        if self.path == "/api/checkout":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    def do_POST(self):
        # ── Stripe checkout endpoint ──
        if self.path == "/api/checkout":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = _loads_json(body)
            except (json.JSONDecodeError, ValueError):
                self._json(400, {"error": "Invalid JSON"})
                return
            items = req.get("items", [])
            if not items:
                self._json(400, {"error": "Cart is empty"})
                return
            if not STRIPE_SECRET_KEY:
                self._json(503, {"error": "Stripe not configured. Set STRIPE_SECRET_KEY env var."})
                return
            try:
                resp = _stripe_create_checkout(
                    items,
                    req.get("successUrl", "https://1bit.systems/store/success"),
                    req.get("cancelUrl", "https://1bit.systems/store"),
                )
                if "error" in resp:
                    self._json(400, resp)
                    return
                # Store pending order for webhook fulfillment
                sid = resp.get("id", "")
                if sid:
                    _pending_orders[sid] = {
                        "items": [{"product": it.get("price_data",{}).get("product_data",{}).get("name","Merch"),
                                   "size": it.get("size",""), "qty": it.get("quantity",1),
                                   "price": it.get("price_data",{}).get("unit_amount",0)}
                                  for it in items],
                        "total": sum(it.get("price_data",{}).get("unit_amount",0) * it.get("quantity",1)
                                     for it in items),
                    }
                    _save_pending()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if self.path == "/api/webhook":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            sig = self.headers.get("Stripe-Signature", "")
            result = _handle_stripe_webhook(body, sig)
            if "status" in result and result["status"] >= 400:
                self._json(result.pop("status", 400), result)
            else:
                self._json(200, result)
            return

        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = _loads_json(body)
            except (json.JSONDecodeError, ValueError):
                self._json(400, {"error": "Invalid JSON"})
                return

            model = req.get("model", "unknown")
            messages = req.get("messages", [])
            stream = req.get("stream", False)
            extra_kwargs = {k: v for k, v in req.items() if k not in ("model", "messages", "stream")}
            print(f"  REQ model={model!r} msgs={len(messages)} stream={stream} extra_keys={list(extra_kwargs.keys())}", flush=True)

            # Streaming requested but not (yet) truly supported — just respond
            # non-streaming. Most clients handle this gracefully.
            _ = stream  # consumed, ignored

            # Route — npu:// bypasses size estimation
            if model.startswith("npu://"):
                device = "npu"
                model_size = 0.6  # Qwen3-0.6B
            else:
                model_size = estimate_model_size(model)
                device = select_device(model_size)

            try:
                if device == "npu":
                    resp = backends["npu"].chat(model, messages, **extra_kwargs)
                elif device == "gpu":
                    resp = backends["gpu"].chat(model, messages, **extra_kwargs)
                else:
                    resp = backends["gpu"].chat(model, messages, **extra_kwargs)

                # Tag response with routing info
                if isinstance(resp, dict):
                    resp["x-device"] = device
                    resp["x-model-size"] = f"{model_size}B"
                    if "error" in resp:
                        print(f"  RESP error: {resp['error']}", flush=True)
                    else:
                        txt = resp.get("choices",[{}])[0].get("message",{}).get("content","")[:50]
                        print(f"  RESP ok: {txt!r}", flush=True)

                self._json(200, resp)

            except urllib.error.HTTPError as e:
                self._json(e.code, {"error": e.reason, "x-device": device})
            except Exception as e:
                self._json(502, {"error": str(e), "x-device": device})
        elif self.path == "/v1/batch/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = _loads_json(body)
            except (json.JSONDecodeError, ValueError):
                self._json(400, {"error": "Invalid JSON"})
                return
            try:
                tokens = _parse_token_list(req.get("tokens", []))
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            if not tokens:
                self._json(400, {"error": "No tokens in request"})
                return
            if len(tokens) > 256:
                self._json(400, {"error": "Max 256 tokens"})
                return
            model_path = req.get("model",
                "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx")
            import subprocess
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = (
                "/home/bcloud/torch2aie/toolchain/mlir_aie.libs:"
                "/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64:" +
                env.get("LD_LIBRARY_PATH", ""))
            engine = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt"
            try:
                proc = subprocess.run([engine, model_path] + [str(t) for t in tokens],
                    capture_output=True, text=True, env=env, timeout=600)
                stderr_lines = proc.stderr.strip().splitlines()
                tokens_out = []; total_ms = 0
                for line in stderr_lines:
                    if "===" in line:
                        parts = line.strip().split(",")
                        for p in parts:
                            if "ms" in p:
                                total_ms = float(p.split()[0].rstrip('ms'))
                if not math.isfinite(total_ms):
                    total_ms = 0
                import re
                for line in stderr_lines:
                    m = re.search(r'token\s+(\d+).*top8=\[([^\]]+)\]', line)
                    if m:
                        top8 = [int(x) for x in m.group(2).split(",")]
                        tokens_out.append({"index": int(m.group(1)), "top8": top8})
                self._json(200, {
                    "object": "batch.completion", "tokens": tokens_out,
                    "total_ms": total_ms,
                    "ms_per_token": total_ms / len(tokens) if tokens else 0,
                    "x-tokens": len(tokens), "x-device": "npu"})
            except subprocess.TimeoutExpired:
                self._json(504, {"error": "Engine timeout"})
            except Exception as e:
                self._json(502, {"error": str(e)})
        else:
            self._json(404, {"error": "Not found"})

    def _serve_store(self):
        store_path = os.path.join(os.path.dirname(__file__), "..", "site", "store", "index.html")
        try:
            with open(store_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._json(404, {"error": "Store not found"})

    def _json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        method, path, code = args[0], args[1], args[2]
        device = self.headers.get("X-Device", "?")
        print(f"[{time.strftime('%H:%M:%S')}] {method} {path} → {code}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NPU+GPU+CPU Control Plane Daemon")
    parser.add_argument("--port", type=int, default=9090, help="Gateway port")
    parser.add_argument("--npu-port", type=int, default=52625, help="NPU backend port")
    parser.add_argument("--gpu-port", type=int, default=13305, help="GPU backend port")
    parser.add_argument("--no-auto", action="store_true", help="Don't auto-start backends")
    args = parser.parse_args()

    global backends
    backends = {
        "npu": NPUBackend(port=args.npu_port),
        "gpu": GPUBackend(port=args.gpu_port),
    }

    print("=" * 60)
    print("  NPU + GPU + CPU = Unified Control Plane")
    print("  Gateway: http://0.0.0.0:{}".format(args.port))
    print("=" * 60)
    print()

    if not args.no_auto:
        print("Starting backends...")
        backends["npu"].start()
        # GPU backend is optional — skip if lemond not found
        try:
            backends["gpu"].start()
        except FileNotFoundError:
            print("  ⚠️  lemond not found — GPU backend disabled")
        print()

    # Signal handling for clean shutdown
    def shutdown(sig, frame):
        print("\nShutting down...")
        backends["npu"].stop()
        backends["gpu"].stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Periodic zombie reaper — lemond GPU backend often dies immediately on
    # systems without ROCm, leaving a defunct process. This timer reaps it.
    def _reaper():
        while True:
            try:
                os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pass  # no children to reap
            threading.Event().wait(5)

    _reaper_thread = threading.Thread(target=_reaper, daemon=True)
    _reaper_thread.start()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Gateway listening on http://0.0.0.0:{args.port}")
    print(f"  GET  /v1/health     — Device and backend status")
    print(f"  GET  /v1/models     — List available models")
    print(f"  POST /v1/chat/completions — Route to NPU/GPU/CPU")
    print(f"  POST /v1/batch/completions — Batch prefill (NPU multi-token)")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()
