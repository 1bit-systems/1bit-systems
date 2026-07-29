// jarvis_server.cpp — local private AI agent server.
// C++ port of jarvis/server.py + ui.py + tools.py + rag.py + planner.py +
// routing.py + beacon.py, all recovered from git history at commit
// c252174aa~1 (jarvis/ was deleted by c252174aa, "Remove all Python -
// C++ only", as collateral damage — this restores its actual agent-level
// behavior in the mandated language). See
// .claude/plans/swirling-tumbling-frost.md for the full port plan.
//
// Architecture: a separate process from unified_server (port 8088), talked
// to over HTTP for actual model inference — not a second in-process
// BackendManager. Own port 8080, matching the original design and the
// beacon/1bit-Mobile pairing contract.
//
// Phase 1: orchestration — chat, RAG, tools, planner, session memory,
// permission gate, beacon, UI.
// Phase 2 (this file): STT via the native src/whisper.cpp — see
// .claude/plans/swirling-tumbling-frost.md. Real, correct, but slow: this
// project's whisper.cpp is a from-scratch scalar CPU reference
// implementation (kernels/whisper_kernels.hip exist for GPU accel but
// are not wired into the forward pass) — a single 30s-context tiny.en
// pass did not finish in 10 minutes in this environment. Usable for
// correctness verification with short/patient testing, not yet a snappy
// live demo. TTS is Phase 3, not in this file.
//
// Build: cmake --build . --target jarvis_server

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <unistd.h>
#include <sys/wait.h>

#include "jarvis/audio_out.h"
#include "jarvis/audio_stream.h"
#include "jarvis/beacon.h"
#include "jarvis/codec_tts.h"
#include "jarvis/context.h"
#include "jarvis/persona.h"
#include "jarvis/planner.h"
#include "jarvis/rag.h"
#include "jarvis/routing.h"
#include "jarvis/tools.h"
#include "jarvis/tts.h"
#include "jarvis/vad.h"
#include "jarvis/auth.h"
#include "jarvis/usage.h"
#include "jarvis/billing.h"
#include "whisper.h"

using json = nlohmann::json;
using namespace jarvis;

// ── WebSocket audio stream server (runs on separate port) ────────────
//
// httplib v0.18.1 does not include built-in WebSocket support, so we
// run a lightweight WebSocket server on a separate background thread.
// The main HTTP server exposes /v1/audio/stream as an HTTP streaming
// alternative and /v1/audio/stream/info for WebSocket connection info.
static std::unique_ptr<WebSocketServer> g_ws_server;

// ── UI (exact port of jarvis/ui.py's CHAT_HTML) ──────────────────────────
static const char* CHAT_HTML = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JARVIS</title>
<style>
:root{--bg:#021621;--panel:#06222f;--border:#0e3346;--text:#e7f6fd;--muted:#86adbf;--green:#00ff00;--pink:#f00fd2;--blue:#12a0ed;--mono:'JetBrains Mono',monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--border);background:rgba(2,22,33,.9)}
header .logo{font-family:var(--mono);font-weight:800;font-size:18px;color:var(--green)}
header .logo .p{color:var(--pink)}
.chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:82%;padding:10px 14px;border-radius:12px;line-height:1.5;font-size:14px;white-space:pre-wrap}
.msg.user{background:var(--blue);align-self:flex-end}
.msg.assistant{background:var(--panel);border:1px solid var(--border);align-self:flex-start}
.msg.system{background:var(--pink);color:var(--bg);align-self:center;font-size:11px;padding:5px 12px;border-radius:999px;font-weight:600}
.msg .label{font-size:9px;color:var(--muted);margin-bottom:3px;font-weight:700;text-transform:uppercase}
.msg img{max-width:100%;border-radius:8px;margin-top:6px;max-height:200px}
.msg .tts-btn{font-size:11px;color:var(--green);cursor:pointer;margin-top:4px;padding:2px 8px;border:1px solid var(--green);border-radius:6px;background:transparent}
.msg .tts-btn:hover{background:rgba(0,255,0,.1)}
.input-row{display:flex;gap:8px;padding:10px 16px;border-top:1px solid var(--border);background:rgba(2,22,33,.95);align-items:center}
.input-row input{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 14px;color:var(--text);font-size:13px;outline:none}
.input-row input:focus{border-color:var(--green)}
.input-row .btn{background:var(--green);color:var(--bg);border:none;border-radius:8px;padding:10px 14px;font-weight:700;cursor:pointer;font-size:13px}
.input-row .btn:hover{box-shadow:0 0 16px rgba(0,255,0,.2)}
.input-row .btn:disabled{opacity:.4}
.input-row .btn-icon{width:36px;height:36px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;display:inline-flex;align-items:center;justify-content:center;font-size:16px}
.input-row .btn-icon:hover{border-color:var(--green);color:var(--green)}
.input-row .btn-icon.recording{background:var(--pink);color:var(--bg);border-color:var(--pink);animation:pulse 1s infinite}
@keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.05)}100%{transform:scale(1)}}
select{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:6px 8px;color:var(--text);font-size:11px;outline:none;cursor:pointer}
.typing{color:var(--muted);font-style:italic;font-size:13px}
</style>
</head>
<body>
<header>
  <div class="logo"><span style=color:var(--green)>J</span>AR<span class=p>V</span>IS</div>
  <select id="model-select" onchange="switchModel(this.value)">
    <option value=auto>Auto</option>
    <option value=qwen3:0.6b>NPU</option>
    <option value=qwen3vl:4b>Vision</option>
    <option value=qwen3.5:9b>GPU</option>
  </select>
  <div style="margin-left:auto;font-size:10px;color:var(--muted)">
    <span id=npu-status>NPU</span>
  </div>
</header>
<div class=chat id=chat>
  <div class="msg system">JARVIS ready. Chat, vision, voice, RAG.</div>
</div>
<div class=input-row>
  <button class=btn-icon id=mic-btn onclick=toggleMic() title="Voice">&#127908;</button>
  <button class=btn-icon onclick="document.getElementById('img-input').click()" title="Image">&#128247;</button>
  <input type=file id=img-input accept="image/*" style=display:none onchange=uploadImage(this)>
  <input id=input type=text placeholder="Message JARVIS..." autofocus
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}">
  <button class=btn id=send-btn onclick=send()>Send</button>
</div>
<audio id=tts-audio style=display:none></audio>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),btn=document.getElementById('send-btn'),micBtn=document.getElementById('mic-btn'),ttsAudio=document.getElementById('tts-audio');
let mode='auto',streaming=false,mediaRecorder=null,audioChunks=[],recording=false,uploadedImage=null;
function switchModel(v){mode=v}
function addMsg(role,content,opts){
  opts=opts||{};const d=document.createElement('div');d.className='msg '+role;
  if(role==='assistant'){const l=document.createElement('div');l.className='label';l.textContent=opts.model||'JARVIS';d.appendChild(l)}
  if(opts.image){const i=document.createElement('img');i.src=opts.image;d.appendChild(i)}
  const t=document.createElement('div');t.textContent=content;d.appendChild(t);
  if(role==='assistant'&&content){const b=document.createElement('button');b.className='tts-btn';b.textContent='🔊';b.onclick=()=>speak(content);d.appendChild(b)}
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight
}
function addTyping(){const d=document.createElement('div');d.className='msg assistant typing';d.id='typing';d.textContent='...';chat.appendChild(d)}
function removeTyping(){const t=document.getElementById('typing');if(t)t.remove()}
function uploadImage(el){const f=el.files[0];if(!f)return;const r=new FileReader();r.onload=function(e){uploadedImage=e.target.result;addMsg('user','[Image]',{image:uploadedImage})};r.readAsDataURL(f);el.value=''}
async function speak(t){try{const r=await fetch('/v1/audio/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:t})});if(!r.ok)return;ttsAudio.src=URL.createObjectURL(await r.blob());ttsAudio.play()}catch(e){}}
async function toggleMic(){if(recording){mediaRecorder.stop();recording=false;micBtn.classList.remove('recording');return}
try{const s=await navigator.mediaDevices.getUserMedia({audio:true});mediaRecorder=new MediaRecorder(s,{mimeType:'audio/webm'});audioChunks=[];mediaRecorder.ondataavailable=e=>audioChunks.push(e.data);mediaRecorder.onstop=async()=>{const b=new Blob(audioChunks,{type:'audio/webm'});const f=new FormData();f.append('file',b,'audio.webm');addMsg('user','🎤');addTyping();try{const r=await fetch('/v1/audio/transcriptions',{method:'POST',body:f});const d=await r.json();removeTyping();const t=d.text||'';if(t&&t!=='[silence]'){input.value=t;send()}}catch(e){removeTyping()}s.getTracks().forEach(t=>t.stop())};mediaRecorder.start();recording=true;micBtn.classList.add('recording')}catch(e){}}
async function send(){let t=input.value.trim();if(!t&&!uploadedImage)return
input.value='';addMsg('user',t);addTyping();streaming=true;btn.disabled=true
try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mode!=='auto'?mode:undefined,messages:[{role:'user',content:t}],max_tokens:256,stream:true})});removeTyping();await handleStream(r)}catch(e){removeTyping();addMsg('system','Error')}
streaming=false;btn.disabled=false}
async function handleStream(resp){const r=resp.body.getReader(),d=new TextDecoder();let full='',first=true,m=''
while(true){const{done,value}=await r.read();if(done)break;const c=d.decode(value);for(const l of c.split('\n')){if(!l.startsWith('data: '))continue;const s=l.slice(6);if(s==='[DONE]')continue;try{const j=JSON.parse(s);const x=j.choices?.[0]?.delta?.content||'';if(x)full+=x;if(j.model)m=j.model;if(first&&full){const div=document.createElement('div');div.className='msg assistant';const lbl=document.createElement('div');lbl.className='label';lbl.textContent=m||'JARVIS';div.appendChild(lbl);const txt=document.createElement('div');txt.textContent=full;div.appendChild(txt);const tb=document.createElement('button');tb.className='tts-btn';tb.textContent='🔊';tb.onclick=()=>speak(full);div.appendChild(tb);div.id='last-msg';chat.appendChild(div);first=false}else if(!first&&full){const e=document.getElementById('last-msg');if(e)e.querySelector('div:nth-child(2)').textContent=full}
chat.scrollTop=chat.scrollHeight}catch(e){}}}}
</script>
</body>
</html>)HTML";

// ── Global state ──────────────────────────────────────────────────────
static KnowledgeBase g_kb;
static int g_port = 8080;
static jarvis::CodecTts g_codec_tts;

// ── Co-Host Intelligence (Phase 2.2) ─────────────────────────────────
static jarvis::PersonaManager g_persona_mgr;
static jarvis::ContextMemory g_context_mem(50);

// ── Commercial API / SaaS Layer (Phase 2.3) ─────────────────────────
static jarvis::AuthManager g_auth_mgr;
static jarvis::UsageTracker g_usage_tracker;
static jarvis::BillingManager g_billing_mgr;

// Thread-local current owner, set by auth pre-routing handler
static thread_local std::string tls_current_owner;
static const std::string& current_owner() { return tls_current_owner; }

// PlanTier string conversion helpers
static const char* plan_tier_to_string(jarvis::PlanTier tier) {
    using jarvis::PlanTier;
    switch (tier) {
        case PlanTier::FREE:       return "free";
        case PlanTier::BASIC:      return "basic";
        case PlanTier::PRO:        return "pro";
        case PlanTier::ENTERPRISE: return "enterprise";
        case PlanTier::CUSTOM:     return "custom";
    }
    return "free";
}

static jarvis::PlanTier string_to_plan_tier(const std::string& s) {
    using jarvis::PlanTier;
    if (s == "basic")      return PlanTier::BASIC;
    if (s == "pro")        return PlanTier::PRO;
    if (s == "enterprise") return PlanTier::ENTERPRISE;
    if (s == "custom")     return PlanTier::CUSTOM;
    return PlanTier::FREE;
}

// ── Whisper STT (lazy singleton, matches the original's threading.Lock-
// guarded lazy load in the deleted jarvis/stt.py) ─────────────────────
static WhisperModel* g_whisper = nullptr;
static std::mutex g_whisper_mutex;

static WhisperModel* get_whisper_model() {
    std::lock_guard<std::mutex> lock(g_whisper_mutex);
    if (g_whisper) return g_whisper;
    const char* path = getenv("WHISPER_MODEL_PATH");
    if (!path || !*path) return nullptr;
    auto* m = new WhisperModel();
    if (!m->load_from_gguf(path)) {
        delete m;
        return nullptr;
    }
    g_whisper = m;
    return g_whisper;
}

// ── Helpers ────────────────────────────────────────────────────────────

// Flattens a message's `content` field (string, or array of {type,text}
// parts per the OpenAI content-parts convention) to plain text.
static std::string flatten_content(const json& content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {
        std::string out;
        for (auto& part : content) {
            if (part.is_object() && part.value("type", "") == "text") out += part.value("text", "");
        }
        return out;
    }
    return "";
}

static bool content_has_image(const json& content) {
    if (!content.is_array()) return false;
    for (auto& part : content) {
        if (part.is_object() && part.value("type", "") == "image_url") return true;
    }
    return false;
}

// Auto model selection: any image part -> vision; else short text -> the
// fast default, longer text -> the bigger model. Matches the original's
// thresholds exactly (< 500 chars).
static std::string auto_select_model(const json& messages) {
    size_t total_chars = 0;
    for (auto& msg : messages) {
        if (msg.contains("content")) {
            if (content_has_image(msg["content"])) return "qwen3vl:4b";
            if (msg["content"].is_string()) total_chars += msg["content"].get<std::string>().size();
        }
    }
    return total_chars < 500 ? "qwen3:0.6b" : "qwen3.5:9b";
}

static bool any_system_mentions_tool_call(const json& messages) {
    for (auto& msg : messages) {
        if (msg.value("role", "") == "system") {
            std::string c = flatten_content(msg.value("content", json("")));
            if (c.find("TOOL_CALL") != std::string::npos) return true;
        }
    }
    return false;
}

// Runs the tool-call loop against `content` (a model's raw reply). Returns
// the final grounded text and, if a call happened, a small info block.
struct ChatFn {
    std::function<json(const std::string&, const json&, int, float)> fn;
};

// Check whether the model's answer appears grounded in the tool result.
// Uses a simple heuristic: if the tool result has content, the answer must
// contain at least some of the same tokens to pass. Otherwise the result
// is prepended as a "source" prefix so the user gets the real data.
static std::string check_tool_grounding(const std::string& answer, const json& result, const std::string& tool_name) {
    // Only validate search_knowledge results — other tools (get_time, list_models, add_note)
    // have trivial results where grounding checks aren't meaningful.
    if (tool_name != "search_knowledge") return answer;

    // Extract all result-snippets into a single string
    std::string result_text;
    if (result.contains("results") && result["results"].is_array()) {
        for (auto& r : result["results"]) {
            if (r.contains("snippet")) result_text += r["snippet"].get<std::string>() + " ";
        }
    }

    // Strip whitespace and check for emptiness
    auto trim = [](const std::string& s) -> std::string {
        size_t a = s.find_first_not_of(" \n\r\t");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \n\r\t");
        return s.substr(a, b - a + 1);
    };

    std::string trimmed = trim(result_text);
    if (trimmed.empty()) return answer;  // no result content to ground against

    // Try to find at least one significant word from the result in the answer.
    // Split result into words, skip short/trivial ones, check each.
    std::string candidate;
    for (char c : trimmed) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.') candidate += c;
        else if (!candidate.empty()) {
            if (candidate.size() >= 4) {
                if (answer.find(candidate) != std::string::npos) {
                    return answer;  // grounded — found a term from the result
                } else if (answer.find(candidate + " ") != std::string::npos ||
                           answer.find(candidate + ".") != std::string::npos ||
                           answer.find(candidate + ",") != std::string::npos) {
                    return answer;  // grounded with punctuation boundary
                }
            }
            candidate.clear();
        }
    }
    // Check the last candidate too
    if (candidate.size() >= 4 && answer.find(candidate) == std::string::npos) {
        candidate.clear();
    }

    // If no grounding found and answer isn't a simple "I couldn't find..." / denial, prepend result
    std::string lower = answer;
    for (auto& c : lower) c = tolower(c);
    if (lower.find("could not") != std::string::npos ||
        lower.find("couldn't") != std::string::npos ||
        lower.find("not found") != std::string::npos ||
        lower.find("no result") != std::string::npos ||
        lower.find("no information") != std::string::npos) {
        return answer;  // legitimate "not found" response
    }

    // Answer appears to ignore the tool result — prepend the actual data
    std::string grounded = "[Based on knowledge base lookup:]\n\n";
    for (auto& r : result["results"]) {
        if (r.contains("path")) grounded += "Source: " + r["path"].get<std::string>() + "\n";
        if (r.contains("snippet")) grounded += r["snippet"].get<std::string>() + "\n\n";
    }
    grounded += "---\n\n" + answer;
    return grounded;
}

static std::pair<std::string, json> resolve_tool_call(const ChatFn& chat_fn, const std::string& model_target,
                                                        json messages, const std::string& content, int max_tokens,
                                                        float temperature, bool allow_write) {
    auto call = parse_tool_call(content);
    if (!call) return {content, json()};

    ToolRunResult tr = run_tool(g_kb, call->name, call->arguments, allow_write);
    messages.push_back({{"role", "assistant"}, {"content", content}});
    messages.push_back({{"role", "user"}, {"content", format_tool_followup(tr.result, tr.allowed)}});

    json follow_up = chat_fn.fn(model_target, messages, max_tokens, temperature);
    std::string final_text;
    if (follow_up.contains("choices") && !follow_up["choices"].empty())
        final_text = follow_up["choices"][0]["message"].value("content", "");
    else if (follow_up.contains("response"))
        final_text = follow_up.value("response", "");
    else
        final_text = content; // backend error — surface the original reply rather than nothing

    // Grounding check: if the model ignored the tool result, surface the real data
    if (tr.allowed && !tr.result.is_null()) {
        final_text = check_tool_grounding(final_text, tr.result, call->name);
    }

    json tool_info = {{"name", call->name}, {"arguments", call->arguments}, {"allowed", tr.allowed}, {"result", tr.result}};
    return {final_text, tool_info};
}

// ── /v1/chat/completions (+ /api/chat alias) ──────────────────────────
static json handle_chat(const json& body) {
    json messages = body.value("messages", json::array());
    int max_tokens = body.value("max_tokens", 256);
    float temperature = body.value("temperature", 0.7f);
    bool use_rag = body.value("rag", true);
    std::string session_id = body.value("session_id", "default");
    bool use_tools = body.value("tools", true);
    bool allow_write = body.value("allow_write", false);
    bool stream = body.value("stream", false);
    if (stream) use_tools = false; // tool-call loop needs the full text, not a stream

    // Server-side session memory, layered underneath whatever history the
    // client itself sent.
    auto history = g_kb.get_recent_conversation(session_id);
    json full_messages = json::array();
    for (auto& turn : history) full_messages.push_back({{"role", turn.role}, {"content", turn.content}});
    for (auto& m : messages) full_messages.push_back(m);

    // Find + persist the last user turn before a reply exists.
    std::string last_user_msg;
    int last_user_idx = -1;
    for (int i = (int)full_messages.size() - 1; i >= 0; i--) {
        if (full_messages[i].value("role", "") == "user") {
            last_user_msg = flatten_content(full_messages[i].value("content", json("")));
            last_user_idx = i;
            break;
        }
    }
    if (last_user_idx >= 0) {
        g_kb.save_turn(session_id, "user", last_user_msg);
        g_context_mem.add_turn("user", last_user_msg);
    }

    // Build persona system prompt and prepend before anything else.
    {
        std::string persona_prompt = g_persona_mgr.build_system_prompt();
        if (!persona_prompt.empty()) {
            json sys = {{"role", "system"}, {"content", persona_prompt}};
            full_messages.insert(full_messages.begin(), sys);
            if (last_user_idx >= 0) last_user_idx++;
        }
    }

    // Layer context memory: inject recent conversation history as a system message.
    {
        std::string ctx = g_context_mem.build_context(5);
        if (!ctx.empty()) {
            json sys = {{"role", "system"}, {"content", ctx}};
            full_messages.insert(full_messages.begin(), sys);
            if (last_user_idx >= 0) last_user_idx++;
        }
    }

    if (use_tools && !any_system_mentions_tool_call(full_messages)) {
        json sys = {{"role", "system"}, {"content", SYSTEM_PROMPT_TOOLS}};
        full_messages.insert(full_messages.begin(), sys);
        if (last_user_idx >= 0) last_user_idx++;
    }

    std::string model_id = body.value("model", "");
    if (model_id.empty() || model_id == "auto") model_id = auto_select_model(full_messages);

    // RAG: inject context into the last user message's content.
    if (use_rag && last_user_idx >= 0 && !last_user_msg.empty()) {
        std::string ctx = g_kb.get_knowledge_context(last_user_msg);
        if (!ctx.empty()) {
            full_messages[last_user_idx]["content"] = ctx + "\n\nQ: " + last_user_msg;
        }
    }

    Route route = resolve_model(model_id);

    std::string final_text;
    json tool_info;

    if (route.backend == RouteBackend::Vision) {
        // Vision: flatten all message text (images are not actually sent
        // to the model in this port either — see plan's "deliberately not
        // preserved" note; this mirrors the original's incomplete vision
        // path, not a regression introduced here).
        std::string vision_prompt;
        for (auto& m : full_messages) vision_prompt += flatten_content(m.value("content", json("")));
        if (vision_prompt.empty()) vision_prompt = "Describe this image.";
        json vmsgs = json::array({{{"role", "user"}, {"content", vision_prompt}}});
        json result = unified_chat(route.target_model, vmsgs, max_tokens, temperature);
        if (result.contains("choices") && !result["choices"].empty())
            final_text = result["choices"][0]["message"].value("content", "");
        else
            final_text = "[error: " + result.value("error", "vision backend failed") + "]";
        // No tool loop, no session save for vision — matches original.
    } else if (route.backend == RouteBackend::Ollama) {
        ChatFn fn{[](const std::string& m, const json& msgs, int mt, float t) { return ollama_chat(m, msgs, mt, t); }};
        json result = fn.fn(route.target_model, full_messages, max_tokens, temperature);
        std::string content = result.value("response", "");
        if (result.contains("error")) content = "[error: " + result["error"].get<std::string>() + "]";
        if (use_tools) {
            auto [text, info] = resolve_tool_call(fn, route.target_model, full_messages, content, max_tokens, temperature, allow_write);
            final_text = text;
            tool_info = info;
        } else {
            final_text = content;
        }
        g_kb.save_turn(session_id, "assistant", final_text);
        g_context_mem.add_turn("assistant", final_text);
        // Track usage (estimate ~4 chars per token)
        g_usage_tracker.record_usage(current_owner(), 0.0,
                                      (int64_t)(final_text.size() / 4));
    } else {
        ChatFn fn{[](const std::string& m, const json& msgs, int mt, float t) { return unified_chat(m, msgs, mt, t); }};
        json result = fn.fn(route.target_model, full_messages, max_tokens, temperature);
        std::string content;
        if (result.contains("choices") && !result["choices"].empty())
            content = result["choices"][0]["message"].value("content", "");
        else
            content = "[error: " + result.value("error", "unified backend failed") + "]";
        if (use_tools) {
            auto [text, info] = resolve_tool_call(fn, route.target_model, full_messages, content, max_tokens, temperature, allow_write);
            final_text = text;
            tool_info = info;
        } else {
            final_text = content;
        }
        g_kb.save_turn(session_id, "assistant", final_text);
        g_context_mem.add_turn("assistant", final_text);
        // Track usage (estimate ~4 chars per token)
        g_usage_tracker.record_usage(current_owner(), 0.0,
                                      (int64_t)(final_text.size() / 4));
    }

    json response = {
        {"id", "c-" + std::to_string((unsigned)time(nullptr))},
        {"object", "chat.completion"},
        {"model", model_id},
        {"choices", json::array({{{"index", 0},
                                   {"message", {{"role", "assistant"}, {"content", final_text}}},
                                   {"finish_reason", "stop"}}})},
    };
    if (!tool_info.is_null()) response["tool_call"] = tool_info;
    return response;
}

// ── main ───────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    bool no_beacon = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) g_port = atoi(argv[++i]);
        else if (arg == "--no-beacon") no_beacon = true;
    }

    std::string bind_addr = "127.0.0.1";
    if (const char* env = getenv("JARVIS_BIND_ADDR")) bind_addr = env;
    if (bind_addr != "127.0.0.1" && bind_addr != "localhost")
        fprintf(stderr, "WARNING: binding to %s — ensure your firewall is configured.\n", bind_addr.c_str());

    httplib::Server svr;
    svr.set_payload_max_length(16 * 1024 * 1024); // 16 MiB, matches original MAX_BODY_SIZE

    // ── Auth middleware + CORS as pre-routing handler ────────────────
    // All /v1/chat/*, /v1/audio/*, /v1/voice/*, /v1/api-key/*,
    // /v1/usage, /v1/billing/* require API key auth.
    // Free tier: /, /chat, /health, /live, /v1/models, /v1/pricing,
    // /v1/billing/webhook, /v1/audio/devices — public, no auth.
    static const std::string kAllowedOrigin = "http://127.0.0.1";
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        // CORS preflight
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Origin", kAllowedOrigin);
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }

        // Public endpoints: no auth required
        std::string path = req.path;
        bool public_path = (path == "/" || path == "/chat" ||
            path == "/health" || path == "/live" ||
            path == "/v1/models" ||
            path == "/v1/pricing" ||
            path == "/v1/billing/webhook" ||
            path.rfind("/v1/audio/devices", 0) == 0);
        if (public_path) {
            return httplib::Server::HandlerResponse::Unhandled;
        }

        // Auth-protected paths:
        bool protected_path = (path.rfind("/v1/chat", 0) == 0 ||
            path.rfind("/v1/audio/", 0) == 0 ||
            path.rfind("/v1/voice/", 0) == 0 ||
            path.rfind("/v1/api-key", 0) == 0 ||
            path == "/v1/usage" ||
            path.rfind("/v1/billing/", 0) == 0);

        if (protected_path) {
            auto auth_it = req.headers.find("Authorization");
            if (auth_it == req.headers.end()) {
                res.status = 401;
                res.set_content(json{{"error", "missing Authorization header"}}.dump(), "application/json");
                res.set_header("Access-Control-Allow-Origin", kAllowedOrigin);
                return httplib::Server::HandlerResponse::Handled;
            }

            const jarvis::ApiKey* key = g_auth_mgr.validate(auth_it->second);
            if (!key) {
                res.status = 401;
                res.set_content(json{{"error", "invalid or expired API key"}}.dump(), "application/json");
                res.set_header("Access-Control-Allow-Origin", kAllowedOrigin);
                return httplib::Server::HandlerResponse::Handled;
            }

            // Rate limit check
            if (!g_auth_mgr.check_rate_limit(key->owner_id)) {
                res.status = 429;
                res.set_content(json{{"error", "rate limit exceeded"}}.dump(), "application/json");
                res.set_header("Access-Control-Allow-Origin", kAllowedOrigin);
                return httplib::Server::HandlerResponse::Handled;
            }

            // Usage limit check — return 402 Payment Required when over limit
            std::string limit_err = g_usage_tracker.check_limits(key->owner_id, key->tier);
            if (!limit_err.empty()) {
                res.status = 402;
                json err_body = {
                    {"error", limit_err},
                    {"upgrade", "Please upgrade your plan at https://zaya.ai/billing"}
                };
                res.set_content(err_body.dump(), "application/json");
                res.set_header("Access-Control-Allow-Origin", kAllowedOrigin);
                return httplib::Server::HandlerResponse::Handled;
            }

            // Store authenticated owner for downstream handlers
            tls_current_owner = key->owner_id;
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });
    auto add_cors = [](httplib::Response& res) { res.set_header("Access-Control-Allow-Origin", kAllowedOrigin); };

    auto serve_ui = [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(CHAT_HTML, "text/html");
        add_cors(res);
    };
    svr.Get("/", serve_ui);
    svr.Get("/chat", serve_ui);

    // ── Dashboard static file server ────────────────────────────────
    svr.set_mount_point("/dashboard", "./site/dashboard");
    svr.Get("/dashboard", [&](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("/dashboard/");
    });

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
        add_cors(res);
    });
    svr.Get("/live", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
        add_cors(res);
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (auto& id : model_ids()) arr.push_back(id);
        res.set_content(json{{"models", arr}}.dump(), "application/json");
        add_cors(res);
    });

    svr.Get("/v1/knowledge", [&](const httplib::Request&, httplib::Response& res) {
        json entries = json::array();
        for (auto& path : g_kb.all_files()) {
            std::ifstream f(path, std::ios::binary);
            std::string title = path.filename().string();
            std::string tags;
            std::string line;
            while (f && std::getline(f, line)) {
                if (line.rfind("# ", 0) == 0) title = line.substr(2);
                else if (line.rfind("tags:", 0) == 0) tags = line.substr(5);
            }
            std::error_code ec;
            std::string rel = std::filesystem::relative(path, g_kb.root(), ec).string();
            std::string category = rel.substr(0, rel.find('/'));
            uintmax_t size = std::filesystem::file_size(path, ec);
            entries.push_back({{"path", rel}, {"title", title}, {"category", category}, {"tags", tags}, {"size", (size_t)size}});
        }
        res.set_content(json{{"entries", entries}}.dump(), "application/json");
        add_cors(res);
    });

    svr.Get("/v1/audio/devices", [&](const httplib::Request&, httplib::Response& res) {
        json devices = json::array();
        PlaybackDevice active;
        bool has_active = find_external_speaker(&active);
        for (auto& d : list_playback_devices()) {
            devices.push_back({{"card", d.card}, {"device", d.device}, {"name", d.name},
                                {"device_name", d.device_name}, {"is_onboard", d.is_onboard}, {"alsa_id", d.alsa_id}});
        }
        json response = {{"devices", devices}, {"external_speaker_connected", has_active}};
        if (has_active) {
            response["active"] = {{"card", active.card}, {"device", active.device}, {"name", active.name},
                                   {"device_name", active.device_name}, {"alsa_id", active.alsa_id}};
        }
        res.set_content(response.dump(2), "application/json");
        add_cors(res);
    });

    auto chat_handler = [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(json{{"error", "Invalid JSON body"}}.dump(), "application/json");
            add_cors(res);
            return;
        }
        json response = handle_chat(body);
        res.set_content(response.dump(2), "application/json");
        add_cors(res);
    };
    svr.Post("/v1/chat/completions", chat_handler);
    svr.Post("/api/chat", chat_handler);

    svr.Post("/v1/knowledge/search", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { body = json::object(); }
        std::string query = body.value("query", "");
        int max_results = body.value("max_results", 5);
        auto results = g_kb.search(query, max_results);
        json arr = json::array();
        for (auto& r : results) arr.push_back({{"path", r.path}, {"title", r.title}, {"score", r.score}, {"snippet", r.snippet}});
        res.set_content(json{{"results", arr}}.dump(), "application/json");
        add_cors(res);
    });

    svr.Post("/v1/knowledge/upload", [&](const httplib::Request& req, httplib::Response& res) {
        std::string filename, content;
        if (req.has_file("file") || req.has_file("filename")) {
            const auto& file = req.has_file("file") ? req.get_file_value("file") : req.get_file_value("filename");
            filename = file.filename;
            content = file.content;
        } else {
            json body;
            try { body = json::parse(req.body); } catch (...) {}
            filename = body.value("filename", "");
            content = body.value("content", "");
        }
        if (filename.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "filename required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }
        std::string path = g_kb.add_document(filename, content);
        res.set_content(json{{"path", path}}.dump(), "application/json");
        add_cors(res);
    });

    // ── Persona endpoints (Phase 2.2) ────────────────────────────────
    svr.Get("/v1/persona", [&](const httplib::Request&, httplib::Response& res) {
        const auto& cfg = g_persona_mgr.active();
        json j;
        j["name"] = cfg.name;
        j["voice_pack"] = cfg.voice_pack;
        j["speaking_style"] = cfg.speaking_style;
        j["speaking_rate"] = cfg.speaking_rate;
        j["voice_pitch"] = cfg.voice_pitch;
        j["enthusiasm"] = cfg.enthusiasm;
        j["formality"] = cfg.formality;
        j["knowledge_domain"] = cfg.knowledge_domain;
        j["catchphrases"] = json::array();
        for (auto& cp : cfg.catchphrases) j["catchphrases"].push_back(cp);
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    svr.Get("/v1/personas", [&](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (auto& name : g_persona_mgr.list_personas()) arr.push_back(name);
        res.set_content(json{{"personas", arr}}.dump(), "application/json");
        add_cors(res);
    });

    svr.Post("/v1/persona", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { body = json::object(); }
        std::string name = body.value("name", "");
        if (name.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "persona name required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }
        if (!g_persona_mgr.set_active(name)) {
            res.status = 404;
            res.set_content(json{{"error", "persona not found: " + name}}.dump(), "application/json");
            add_cors(res);
            return;
        }
        res.set_content(json{{"status", "ok"}, {"persona", name}}.dump(), "application/json");
        add_cors(res);
    });

    svr.Post("/v1/agent/plan", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { body = json::object(); }
        std::string request = body.value("request", "");
        std::string session_id = body.value("session_id", "default");
        bool allow_write = body.value("allow_write", false);

        g_kb.save_turn(session_id, "user", request);
        json result = run_plan(g_kb, request, allow_write);
        g_kb.save_turn(session_id, "assistant", result.value("answer", ""));

        res.set_content(result.dump(2), "application/json");
        add_cors(res);
    });

    svr.Post("/v1/audio/transcriptions", [&](const httplib::Request& req, httplib::Response& res) {
        std::string audio_bytes;
        if (req.has_file("file")) audio_bytes = req.get_file_value("file").content;
        else if (req.has_file("audio")) audio_bytes = req.get_file_value("audio").content;

        if (audio_bytes.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "no audio file"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        WhisperModel* model = get_whisper_model();
        if (!model) {
            res.set_content(json{{"text", "[transcription unavailable: WHISPER_MODEL_PATH not set or model failed to load]"}}.dump(),
                             "application/json");
            add_cors(res);
            return;
        }

        // Normalize to 16kHz mono s16 WAV via ffmpeg — handles whatever
        // format the client actually sent (the UI's mic button records
        // audio/webm, not WAV) and any sample rate; matches the original
        // Python's ffmpeg-based resample step, just applied unconditionally
        // instead of only for non-WAV input.
        std::string tag = std::to_string((long)getpid()) + "_" + std::to_string((long)time(nullptr));
        std::string in_path = "/tmp/jarvis_stt_in_" + tag + ".bin";
        std::string out_path = "/tmp/jarvis_stt_out_" + tag + ".wav";
        {
            std::ofstream f(in_path, std::ios::binary | std::ios::trunc);
            f.write(audio_bytes.data(), (std::streamsize)audio_bytes.size());
        }

        // Paths are server-generated (pid + timestamp, alphanumeric + underscore),
        // but use fork/exec rather than system() to avoid any shell interpretation
        // even if paths somehow contain special characters (issue #964).
        pid_t child = fork();
        int rc = -1;
        if (child == 0) {
            // Child: exec ffmpeg directly, no shell
            execlp("ffmpeg", "ffmpeg", "-y", "-loglevel", "error",
                   "-i", in_path.c_str(),
                   "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                   out_path.c_str(), nullptr);
            _exit(127);  // exec failed
        } else if (child > 0) {
            int status;
            waitpid(child, &status, 0);
            rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }
        std::error_code ec;
        std::filesystem::remove(in_path, ec);

        std::string text;
        double audio_minutes = 0.0;
        if (rc == 0 && std::filesystem::exists(out_path)) {
            int sr = 16000;
            auto pcm = whisper_load_wav(out_path, &sr);
            std::filesystem::remove(out_path, ec);
            if (!pcm.empty()) {
                text = whisper_transcribe(*model, pcm.data(), (int)pcm.size());
                audio_minutes = pcm.size() / (16000.0 * 60.0);
            }
        } else {
            std::filesystem::remove(out_path, ec);
        }
        if (text.empty()) text = "[silence]";

        // Track usage: estimate audio duration from PCM size
        g_usage_tracker.record_usage(current_owner(), audio_minutes, 0);

        res.set_content(json{{"text", text}}.dump(), "application/json");
        add_cors(res);
    });

    // ── /v1/audio/chat : voice-in/voice-out (VAD + Whisper + LLM + TTS) ──
    // Accepts an audio file, detects speech segments via VAD, transcribes
    // via Whisper, runs through LLM with persona + context, and returns
    // synthesized audio + transcript.
    svr.Post("/v1/audio/chat", [&](const httplib::Request& req, httplib::Response& res) {
        // ── Extract audio ─────────────────────────────────────────
        std::string audio_bytes;
        if (req.has_file("file")) audio_bytes = req.get_file_value("file").content;
        else if (req.has_file("audio")) audio_bytes = req.get_file_value("audio").content;

        if (audio_bytes.empty()) {
            json body;
            try { body = json::parse(req.body); } catch (...) {}
            if (body.contains("audio")) {
                // Base64 or raw PCM — treat as base64 for now
                std::string b64 = body["audio"].get<std::string>();
                (void)b64; // placeholder for base64 decode
            }
        }

        if (audio_bytes.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "no audio data"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        std::string session_id = req.get_param_value("session_id");
        if (session_id.empty()) session_id = "default";

        // ── Normalize audio to 16kHz mono S16 WAV via ffmpeg ────────
        std::string tag = std::to_string((long)getpid()) + "_" + std::to_string((long)time(nullptr));
        std::string in_path = "/tmp/jarvis_chat_audio_in_" + tag + ".bin";
        std::string out_path = "/tmp/jarvis_chat_audio_out_" + tag + ".wav";
        {
            std::ofstream f(in_path, std::ios::binary | std::ios::trunc);
            f.write(audio_bytes.data(), (std::streamsize)audio_bytes.size());
        }

        pid_t child = fork();
        int rc = -1;
        if (child == 0) {
            execlp("ffmpeg", "ffmpeg", "-y", "-loglevel", "error",
                   "-i", in_path.c_str(),
                   "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                   out_path.c_str(), nullptr);
            _exit(127);
        } else if (child > 0) {
            int status;
            waitpid(child, &status, 0);
            rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }
        std::error_code ec;
        std::filesystem::remove(in_path, ec);

        // ── VAD: detect speech segments before transcription ──────
        WhisperModel* model = get_whisper_model();
        std::string transcript;
        std::vector<float> full_pcm;
        bool has_speech = false;
        if (rc == 0 && model && std::filesystem::exists(out_path)) {
            int sr = 16000;
            full_pcm = whisper_load_wav(out_path, &sr);
            std::filesystem::remove(out_path, ec);

            if (!full_pcm.empty()) {
                // Run VAD on the PCM to detect speech segments
                VADConfig vad_cfg;
                vad_cfg.sample_rate = sr;
                VAD vad(vad_cfg);
                vad.process(full_pcm.data(), (int)full_pcm.size());

                // If VAD detected speech, use the last utterance for
                // transcription (better to have VAD-purified audio)
                std::vector<float> vad_audio;
                auto last_utt = vad.get_last_utterance();
                if (!last_utt.empty()) {
                    has_speech = true;
                    vad_audio = std::move(last_utt);
                } else if (vad.is_speaking()) {
                    // Still speaking — use the speech buffer
                    has_speech = true;
                    vad_audio = vad.get_speech_buffer();
                }

                // Transcribe the VAD-isolated audio (or full audio as fallback)
                if (has_speech && !vad_audio.empty()) {
                    transcript = whisper_transcribe(*model, vad_audio.data(), (int)vad_audio.size());
                } else {
                    // Fallback: transcribe full audio even without VAD detection
                    transcript = whisper_transcribe(*model, full_pcm.data(), (int)full_pcm.size());
                    if (!transcript.empty() && transcript != "[silence]") has_speech = true;
                }
            }
        } else {
            std::filesystem::remove(out_path, ec);
        }
        if (!has_speech || transcript.empty() || transcript == "[silence]") {
            res.set_content(json{{"text", "[silence]"}, {"audio", ""}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        // ── LLM call with persona + context ────────────────────────
        g_context_mem.add_turn("user", transcript);

        json msgs = json::array();
        // Build system prompt from persona
        std::string sys_prompt = g_persona_mgr.build_system_prompt();
        if (!sys_prompt.empty()) {
            msgs.push_back({{"role", "system"}, {"content", sys_prompt}});
        }
        // Add conversation context
        std::string ctx = g_context_mem.build_context(5);
        if (!ctx.empty()) {
            msgs.push_back({{"role", "system"}, {"content", ctx}});
        }
        msgs.push_back({{"role", "user"}, {"content", transcript}});

        std::string model_id = "qwen3:0.6b"; // fast default for voice
        Route route = resolve_model(model_id);
        json llm_result;
        if (route.backend == RouteBackend::Ollama) {
            llm_result = ollama_chat(route.target_model, msgs, 128, 0.7f);
        } else {
            llm_result = unified_chat(route.target_model, msgs, 128, 0.7f);
        }

        std::string reply;
        if (llm_result.contains("choices") && !llm_result["choices"].empty())
            reply = llm_result["choices"][0]["message"].value("content", "");
        else if (llm_result.contains("response"))
            reply = llm_result.value("response", "");
        else
            reply = "[error: LLM call failed]";

        g_context_mem.add_turn("assistant", reply);

        // Apply persona catchphrases
        reply = g_persona_mgr.apply_catchphrases(reply);

        // ── Synthesize speech ──────────────────────────────────────
        std::string voice = g_persona_mgr.active().voice_pack;
        if (voice.empty()) voice = "en_US-lessac-medium";

        std::string wav;
        if (g_codec_tts.has_voice(voice)) {
            wav = g_codec_tts.synthesize(reply, voice);
        }
        if (wav.empty()) {
            wav = synthesize_speech(reply, voice);
        }

        std::string audio_b64;
        if (!wav.empty()) {
            // Simple hex encoding as placeholder for proper base64
            // A full base64 implementation would go here
            audio_b64 = "[wav:" + std::to_string(wav.size()) + " bytes]";
        }

        // Track usage for this audio chat turn
        double est_minutes = 1.0; // estimate ~1 min per voice turn
        int64_t est_tokens = (int64_t)(reply.size() / 3);
        if (!full_pcm.empty()) {
            est_minutes = full_pcm.size() / (16000.0 * 60.0);
        }
        g_usage_tracker.record_usage(current_owner(), est_minutes, est_tokens);

        res.set_content(json{{"text", reply}, {"audio", audio_b64}}.dump(), "application/json");
        add_cors(res);
    });

    // ── /v1/voice/packs : list available voice packs ──────────────────
    svr.Get("/v1/voice/packs", [&](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (auto& vp : g_codec_tts.list_voice_packs()) {
            arr.push_back({
                {"name", vp.name},
                {"speaker_name", vp.speaker_name},
                {"language", vp.language},
                {"sample_rate", vp.sample_rate},
                {"path", vp.path}
            });
        }
        res.set_content(json{{"voice_packs", arr}}.dump(), "application/json");
        add_cors(res);
    });

    // ── /v1/audio/stream : HTTP chunked streaming audio ────────────────
    //
    // httplib v0.18.1 lacks built-in WebSocket support, so this endpoint
    // uses HTTP chunked transfer encoding for real-time audio streaming.
    // A separate WebSocket server runs on a different port for native
    // WebSocket clients (see ws_server below).
    //
    // Protocol:
    //   Content-Type: application/octet-stream
    //   - First frame: JSON metadata string (length-prefixed with a 4-byte LE
    //     uint32 header)
    //   - Subsequent frames: raw float32 PCM data (4-byte LE uint32 size header
    //     + data)
    //   - Final frame: empty (size=0)
    //
    // Client cancels by disconnecting.
    //
    svr.Get("/v1/audio/stream", [&](const httplib::Request& req, httplib::Response& res) {
        std::string voice = req.get_param_value("voice");
        std::string text = req.get_param_value("text");

        if (voice.empty() || text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "voice and text query params required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        // Check if voice pack exists
        if (!g_codec_tts.has_voice(voice)) {
            res.status = 404;
            res.set_content(json{{"error", "voice pack not found: " + voice}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        // Stream audio via chunked content provider
        res.set_chunked_content_provider("application/octet-stream",
            [&, voice, text](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                // ── Synthesize ─────────────────────────────────
                std::string wav = g_codec_tts.synthesize(text, voice);
                if (wav.empty()) {
                    // Try Piper fallback
                    wav = synthesize_speech(text, voice);
                }
                if (wav.empty()) {
                    sink.done();
                    return true;
                }

                // ── Parse WAV header ────────────────────────────
                if (wav.size() < 44) {
                    sink.done();
                    return true;
                }

                size_t pcm_offset = 0;
                size_t pcm_size = 0;
                size_t data_start = 12;
                while (data_start + 8 <= wav.size()) {
                    uint32_t chunk_size = *(const uint32_t*)(wav.data() + data_start + 4);
                    if (wav.substr(data_start, 4) == "data") {
                        pcm_offset = data_start + 8;
                        pcm_size = (size_t)chunk_size;
                        break;
                    }
                    data_start += 8 + (size_t)chunk_size;
                }

                if (pcm_offset == 0 || pcm_offset >= wav.size()) {
                    sink.done();
                    return true;
                }

                size_t num_samples = pcm_size / 2; // S16LE mono
                const int16_t* s16 = reinterpret_cast<const int16_t*>(wav.data() + pcm_offset);

                // ── Send metadata frame ────────────────────────
                std::string meta = R"({"sample_rate":24000,"channels":1,"format":"float32"})";
                uint32_t meta_len = (uint32_t)meta.size();
                sink.write((const char*)&meta_len, 4);
                sink.write(meta.data(), meta.size());

                // ── Send audio chunks ────────────────────────────
                static constexpr int kFrameSamples = 312; // 13ms @ 24kHz
                size_t sample_offset = 0;
                while (sample_offset < num_samples) {
                    size_t chunk = std::min((size_t)kFrameSamples, num_samples - sample_offset);

                    // Convert S16 to float32
                    std::vector<float> float_buf(chunk);
                    for (size_t i = 0; i < chunk; i++)
                        float_buf[i] = s16[sample_offset + i] / 32768.0f;

                    uint32_t data_len = (uint32_t)(chunk * sizeof(float));
                    sink.write((const char*)&data_len, 4);
                    sink.write((const char*)float_buf.data(), data_len);

                    sample_offset += chunk;

                    // Pace at real-time
                    int sleep_ms = (int)(chunk * 1000 / 24000);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                }

                // ── End frame ───────────────────────────────────
                uint32_t end_marker = 0;
                sink.write((const char*)&end_marker, 4);
                sink.done();
                return true;
            }
        );
        add_cors(res);
    });

    // ── /v1/audio/stream/info : WebSocket server info ───────────────────
    svr.Get("/v1/audio/stream/info", [&](const httplib::Request&, httplib::Response& res) {
        int ws_port = 0;
        if (g_ws_server) ws_port = g_ws_server->port();
        res.set_content(json{{
            {"websocket_port", ws_port},
            {"protocol", "ws"},
            {"path", "/v1/audio/stream"},
            {"sample_rate", 24000},
            {"channels", 1},
            {"format", "float32"},
            {"http_stream", "/v1/audio/stream?voice=X&text=Y"},
        }}.dump(), "application/json");
        add_cors(res);
    });

    // ── /v1/audio/speech : text-to-speech (codec TTS with Piper fallback) ─
    svr.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { body = json::object(); }
        std::string text = body.value("input", "");
        std::string voice = body.value("voice", "en_US-lessac-medium");
        bool play_local = body.value("play_local", true); // opt-out, matches original

        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "input required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        // Try codec TTS first (native ONNX inference)
        std::string wav;
        if (g_codec_tts.has_voice(voice)) {
            wav = g_codec_tts.synthesize(text, voice);
        }

        // Fall back to Piper if codec TTS returned nothing
        if (wav.empty()) {
            wav = synthesize_speech(text, voice);
        }

        if (wav.empty()) {
            res.status = 502;
            res.set_content(json{{"error", "speech synthesis failed (no voice pack, no piper)"}}.dump(),
                             "application/json");
            add_cors(res);
            return;
        }

        if (play_local) play_wav_local(wav); // fire-and-forget, never blocks this response

        // Track TTS usage: estimate from WAV duration
        if (!wav.empty()) {
            // WAV header at bytes 40-43 contains sample rate (24000 default)
            int sample_rate = 24000;
            if (wav.size() >= 44) {
                uint32_t sr = *(const uint32_t*)(wav.data() + 24);
                if (sr > 0) sample_rate = (int)sr;
            }
            size_t data_size = (wav.size() > 44) ? (wav.size() - 44) : 0;
            double est_minutes = data_size / (double)(sample_rate * 2) / 60.0; // S16LE = 2 bytes/sample
            g_usage_tracker.record_usage(current_owner(), est_minutes, 0);
        }

        res.set_content(wav, "audio/wav");
        add_cors(res);
    });

    // ── SaaS / Commercial API endpoints (Phase 2.3) ─────────────────

    // POST /v1/api-key/create — create new API key (requires existing key)
    svr.Post("/v1/api-key/create", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { body = json::object(); }

        std::string owner_id = current_owner();
        if (owner_id.empty()) {
            res.status = 403;
            res.set_content(json{{"error", "authentication required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        std::string tier_str = body.value("tier", "free");
        PlanTier tier = string_to_plan_tier(tier_str);
        int valid_days = body.value("valid_days", 365);

        std::string key = g_auth_mgr.create_key(owner_id, tier, valid_days);
        g_auth_mgr.save_keys("keys.json");

        res.set_content(json{{"key", key}, {"owner_id", owner_id}, {"tier", tier_str}}.dump(), "application/json");
        add_cors(res);
    });

    // POST /v1/api-key/revoke — revoke API key
    svr.Post("/v1/api-key/revoke", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { body = json::object(); }

        std::string key = body.value("key", "");
        if (key.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "key required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        if (!g_auth_mgr.revoke_key(key)) {
            res.status = 404;
            res.set_content(json{{"error", "key not found"}}.dump(), "application/json");
            add_cors(res);
            return;
        }
        g_auth_mgr.save_keys("keys.json");

        res.set_content(json{{"status", "revoked"}}.dump(), "application/json");
        add_cors(res);
    });

    // GET /v1/api-key/list — list my API keys
    svr.Get("/v1/api-key/list", [&](const httplib::Request&, httplib::Response& res) {
        std::string owner_id = current_owner();
        if (owner_id.empty()) {
            res.status = 403;
            res.set_content(json{{"error", "authentication required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        auto keys = g_auth_mgr.list_keys(owner_id);
        json arr = json::array();
        for (auto& ak : keys) {
            arr.push_back({
                {"key", ak.key.substr(0, 12) + "..."},  // masked
                {"owner_id", ak.owner_id},
                {"tier", static_cast<int>(ak.tier)},
                {"active", ak.active},
                {"created_at", ak.created_at},
                {"expires_at", ak.expires_at},
            });
        }
        res.set_content(json{{"keys", arr}}.dump(), "application/json");
        add_cors(res);
    });

    // GET /v1/usage — get current usage for authenticated owner
    svr.Get("/v1/usage", [&](const httplib::Request&, httplib::Response& res) {
        std::string owner_id = current_owner();
        if (owner_id.empty()) {
            res.status = 403;
            res.set_content(json{{"error", "authentication required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        auto usage = g_usage_tracker.get_usage(owner_id);
        json j;
        j["owner_id"] = usage.owner_id;
        j["minutes_used"] = usage.minutes_used;
        j["tokens_processed"] = usage.tokens_processed;
        j["requests_count"] = usage.requests_count;
        j["period_start"] = usage.period_start;
        j["period_end"] = usage.period_end;

        // Include tier limits for context
        auto keys = g_auth_mgr.list_keys(owner_id);
        PlanTier tier = PlanTier::FREE;
        if (!keys.empty()) tier = keys[0].tier;
        auto limits = TierLimits::for_tier(tier);
        j["limits"] = {
            {"max_minutes", limits.max_minutes},
            {"max_tokens", limits.max_tokens},
            {"max_voices", limits.max_voices},
        };

        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // GET /v1/pricing — get pricing info (public, no auth needed)
    svr.Get("/v1/pricing", [&](const httplib::Request&, httplib::Response& res) {
        auto pricing = g_billing_mgr.get_pricing();
        json j;
        j["basic_monthly"] = pricing.basic_monthly;
        j["pro_monthly"] = pricing.pro_monthly;
        j["enterprise_monthly"] = pricing.enterprise_monthly;
        j["voice_clone_fee"] = pricing.voice_clone_fee;
        j["currency"] = "USD";
        j["tiers"] = json::array();
        for (auto tier : {PlanTier::FREE, PlanTier::BASIC, PlanTier::PRO, PlanTier::ENTERPRISE}) {
            auto limits = TierLimits::for_tier(tier);
            j["tiers"].push_back({
                {"name", plan_tier_to_string(tier)},
                {"price_monthly", limits.price_monthly},
                {"max_minutes", limits.max_minutes},
                {"max_tokens", limits.max_tokens},
                {"max_voices", limits.max_voices},
            });
        }
        res.set_content(j.dump(2), "application/json");
        add_cors(res);
    });

    // POST /v1/billing/webhook — Stripe webhook endpoint (no auth, signature verified)
    svr.Post("/v1/billing/webhook", [&](const httplib::Request& req, httplib::Response& res) {
        std::string signature;
        auto sig_it = req.headers.find("Stripe-Signature");
        if (sig_it != req.headers.end()) signature = sig_it->second;

        if (!g_billing_mgr.process_webhook(req.body, signature)) {
            res.status = 400;
            res.set_content(json{{"error", "webhook processing failed"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
        add_cors(res);
    });

    // GET /v1/billing/portal — stub: return Stripe customer portal URL
    svr.Get("/v1/billing/portal", [&](const httplib::Request& req, httplib::Response& res) {
        std::string owner_id = current_owner();
        if (owner_id.empty()) {
            res.status = 403;
            res.set_content(json{{"error", "authentication required"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        // Look up customer Stripe ID for this owner
        std::string customer_id = g_billing_mgr.get_customer_for_owner(owner_id);
        if (customer_id.empty()) {
            // No Stripe customer yet — return generic billing URL
            res.set_content(json{{"url", "https://zaya.ai/billing"}}.dump(), "application/json");
            add_cors(res);
            return;
        }

        // Stub: real portal URL requires Stripe SDK's
        // stripe.billingPortal.sessions.create({customer, return_url}).
        // Return a configurable base URL with the customer_id as query param.
        const char* portal_base = getenv("STRIPE_PORTAL_URL");
        std::string portal_url = portal_base ? portal_base : "https://zaya.ai/billing/portal";
        portal_url += "?customer_id=" + customer_id;
        std::string return_url = req.get_param_value("return_url");
        if (!return_url.empty()) {
            portal_url += "&return_url=" + return_url;
        }

        res.set_content(json{{"url", portal_url}}.dump(), "application/json");
        add_cors(res);
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404) res.set_content(json{{"error", "nf"}}.dump(), "application/json");
    });

    printf("JARVIS @ http://%s:%d/chat\n", bind_addr.c_str(), g_port);
    printf("  unified_server: %s\n", unified_server_url().c_str());
    printf("  ollama:         %s\n", ollama_url().c_str());
    printf("  knowledge base: %s\n", g_kb.root().c_str());

    // Initialise persona manager
    {
        const char* pd = getenv("PERSONAS_DIR");
        std::string dir = pd ? pd : "";
        int n = g_persona_mgr.scan_directory(dir);
        printf("  personas:         %d loaded (active: %s)\n", n,
               g_persona_mgr.active().name.c_str());
        for (auto& name : g_persona_mgr.list_personas())
            printf("    - %s\n", name.c_str());
    }

    // Initialise codec TTS voice pack scanner
    {
        const char* vp_dir = getenv("VOICE_PACKS_DIR");
        if (vp_dir && *vp_dir) g_codec_tts.set_voice_packs_dir(vp_dir);
        g_codec_tts.scan_voice_packs();
        auto packs = g_codec_tts.list_voice_packs();
        if (!packs.empty()) {
            printf("  codec TTS:        %zu voice pack(s) loaded\n", packs.size());
            for (auto& vp : packs)
                printf("    - %s (speaker=%s, lang=%s)\n", vp.name.c_str(), vp.speaker_name.c_str(), vp.language.c_str());
        } else {
            printf("  codec TTS:        no voice packs found in %s\n",
                   vp_dir ? vp_dir : "~/voice-packs");
        }
    }

    // Start WebSocket audio streaming server (separate port for raw WS)
    {
        int ws_port = 8082;
        const char* ws_port_env = getenv("WS_STREAM_PORT");
        if (ws_port_env && *ws_port_env) ws_port = atoi(ws_port_env);

        g_ws_server = std::make_unique<WebSocketServer>();
        int actual_port = g_ws_server->start(ws_port, &g_codec_tts);
        if (actual_port > 0) {
            printf("  WS stream:        ws://127.0.0.1:%d/v1/audio/stream?voice=X&text=Y\n", actual_port);
        } else {
            printf("  WS stream:        FAILED to start\n");
        }
    }

    if (!no_beacon) start_beacon(g_port);

    if (!svr.listen(bind_addr, g_port)) {
        fprintf(stderr, "Failed to start server on %s:%d\n", bind_addr.c_str(), g_port);
        return 1;
    }
    return 0;
}
