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
#include "jarvis/beacon.h"
#include "jarvis/planner.h"
#include "jarvis/rag.h"
#include "jarvis/routing.h"
#include "jarvis/tools.h"
#include "jarvis/tts.h"
#include "whisper.h"

using json = nlohmann::json;
using namespace jarvis;

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
    if (last_user_idx >= 0) g_kb.save_turn(session_id, "user", last_user_msg);

    if (use_tools && !any_system_mentions_tool_call(full_messages)) {
        json sys = {{"role", "system"}, {"content", SYSTEM_PROMPT_TOOLS}};
        full_messages.insert(full_messages.begin(), sys);
        if (last_user_idx >= 0) last_user_idx++; // shifted by the inserted system message
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

    // CORS: scoped to exactly one origin (http://127.0.0.1), not "*" —
    // deliberate in the original, preserved here.
    static const std::string kAllowedOrigin = "http://127.0.0.1";
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Origin", kAllowedOrigin);
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
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
        if (rc == 0 && std::filesystem::exists(out_path)) {
            int sr = 16000;
            auto pcm = whisper_load_wav(out_path, &sr);
            std::filesystem::remove(out_path, ec);
            if (!pcm.empty()) text = whisper_transcribe(*model, pcm.data(), (int)pcm.size());
        } else {
            std::filesystem::remove(out_path, ec);
        }
        if (text.empty()) text = "[silence]";

        res.set_content(json{{"text", text}}.dump(), "application/json");
        add_cors(res);
    });

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

        std::string wav = synthesize_speech(text, voice);
        if (wav.empty()) {
            res.status = 502;
            res.set_content(json{{"error", "speech synthesis failed (voice model missing or piper failed)"}}.dump(),
                             "application/json");
            add_cors(res);
            return;
        }

        if (play_local) play_wav_local(wav); // fire-and-forget, never blocks this response

        res.set_content(wav, "audio/wav");
        add_cors(res);
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404) res.set_content(json{{"error", "nf"}}.dump(), "application/json");
    });

    printf("JARVIS @ http://%s:%d/chat\n", bind_addr.c_str(), g_port);
    printf("  unified_server: %s\n", unified_server_url().c_str());
    printf("  ollama:         %s\n", ollama_url().c_str());
    printf("  knowledge base: %s\n", g_kb.root().c_str());

    if (!no_beacon) start_beacon(g_port);

    if (!svr.listen(bind_addr, g_port)) {
        fprintf(stderr, "Failed to start server on %s:%d\n", bind_addr.c_str(), g_port);
        return 1;
    }
    return 0;
}
