CHAT_HTML = '''<!DOCTYPE html>
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
  <button class=btn-icon id=mic-btn onclick=toggleMic() title="Voice">🎤</button>
  <button class=btn-icon onclick="document.getElementById('img-input').click()" title="Image">🖼</button>
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
  if(role==='assistant'&&content){const b=document.createElement('button');b.className='tts-btn';b.textContent='\ud83d\udd0a';b.onclick=()=>speak(content);d.appendChild(b)}
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight
}
function addTyping(){const d=document.createElement('div');d.className='msg assistant typing';d.id='typing';d.textContent='...';chat.appendChild(d)}
function removeTyping(){const t=document.getElementById('typing');if(t)t.remove()}
function uploadImage(el){const f=el.files[0];if(!f)return;const r=new FileReader();r.onload=function(e){uploadedImage=e.target.result;addMsg('user','[Image]',{image:uploadedImage})};r.readAsDataURL(f);el.value=''}
async function speak(t){try{const r=await fetch('/v1/audio/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:t})});if(!r.ok)return;ttsAudio.src=URL.createObjectURL(await r.blob());ttsAudio.play()}catch(e){}}
async function toggleMic(){if(recording){mediaRecorder.stop();recording=false;micBtn.classList.remove('recording');return}
try{const s=await navigator.mediaDevices.getUserMedia({audio:true});mediaRecorder=new MediaRecorder(s,{mimeType:'audio/webm'});audioChunks=[];mediaRecorder.ondataavailable=e=>audioChunks.push(e.data);mediaRecorder.onstop=async()=>{const b=new Blob(audioChunks,{type:'audio/webm'});const f=new FormData();f.append('file',b,'audio.webm');addMsg('user','\ud83c\udfa4');addTyping();try{const r=await fetch('/v1/audio/transcriptions',{method:'POST',body:f});const d=await r.json();removeTyping();const t=d.text||'';if(t&&t!=='[silence]'){input.value=t;send()}}catch(e){removeTyping()}s.getTracks().forEach(t=>t.stop())};mediaRecorder.start();recording=true;micBtn.classList.add('recording')}catch(e){}}
async function send(){let t=input.value.trim();if(!t&&!uploadedImage)return
input.value='';addMsg('user',t);addTyping();streaming=true;btn.disabled=true
try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mode!=='auto'?mode:undefined,messages:[{role:'user',content:t}],max_tokens:256,stream:true})});removeTyping();await handleStream(r)}catch(e){removeTyping();addMsg('system','Error')}
streaming=false;btn.disabled=false}
async function handleStream(resp){const r=resp.body.getReader(),d=new TextDecoder();let full='',first=true,m=''
while(true){const{done,value}=await r.read();if(done)break;const c=d.decode(value);for(const l of c.split('\n')){if(!l.startsWith('data: '))continue;const s=l.slice(6);if(s==='[DONE]')continue;try{const j=JSON.parse(s);const x=j.choices?.[0]?.delta?.content||'';if(x)full+=x;if(j.model)m=j.model;if(first&&full){const div=document.createElement('div');div.className='msg assistant';const lbl=document.createElement('div');lbl.className='label';lbl.textContent=m||'JARVIS';div.appendChild(lbl);const txt=document.createElement('div');txt.textContent=full;div.appendChild(txt);const tb=document.createElement('button');tb.className='tts-btn';tb.textContent='\ud83d\udd0a';tb.onclick=()=>speak(full);div.appendChild(tb);div.id='last-msg';chat.appendChild(div);first=false}else if(!first&&full){const e=document.getElementById('last-msg');if(e)e.querySelector('div:nth-child(2)').textContent=full}
chat.scrollTop=chat.scrollHeight}catch(e){}}}}
</script>
</body>
</html>'''