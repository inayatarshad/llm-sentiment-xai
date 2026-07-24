"""Dependency-free WSGI app for the Vercel-hosted NLP // CORE demo."""

import json
import os
import urllib.error
import urllib.request

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NLP // CORE</title><style>
*{box-sizing:border-box}body{margin:0;background:#08080e;color:#d8d8ec;font:16px ui-monospace,Consolas,monospace}
main{width:min(820px,92%);margin:6vh auto}h1{font:900 clamp(2.2rem,7vw,4.6rem) Arial;margin:.2rem 0;
background:linear-gradient(90deg,#00ff8c,#b400ff);color:transparent;background-clip:text}p{color:#8585a2;line-height:1.6}
.eyebrow{color:#00ff8c;letter-spacing:.2em;font-size:.75rem}textarea{width:100%;min-height:190px;margin:1.5rem 0 .8rem;
padding:1rem;background:#0d0d18;color:#e8e8ff;border:1px solid #00ff8c55;border-radius:8px;resize:vertical;font:inherit}
button{background:#00ff8c;color:#07130d;border:0;border-radius:6px;padding:.8rem 1.3rem;font:bold .85rem Arial;letter-spacing:.12em;cursor:pointer}
button:disabled{opacity:.5}.status{min-height:2rem;margin:1rem 0;color:#00ff8c}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem}
.card{background:#0d0d18;border:1px solid #b400ff44;border-left:3px solid #b400ff;border-radius:6px;padding:1rem;margin:.7rem 0}
.card span{display:block;color:#b400ff;font-size:.7rem;letter-spacing:.15em;margin-bottom:.5rem}.wide{grid-column:1/-1}
.note{font-size:.78rem}.error{color:#ff6b8a}@media(max-width:620px){.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">EXPLAINABLE SENTIMENT INTELLIGENCE</div><h1>NLP // CORE</h1>
<p>Analyze sentiment, confidence, themes, tone, and the reasoning behind them in one pass.</p>
<textarea id="text" maxlength="12000" placeholder="Paste text to analyze..."></textarea><button id="run">ANALYZE</button>
<div class="status" id="status"></div><section class="grid" id="results" hidden>
<div class="card"><span>SENTIMENT</span><b id="sentiment"></b></div><div class="card"><span>POLARITY</span><b id="polarity"></b></div>
<div class="card"><span>CONFIDENCE</span><b id="confidence"></b></div><div class="card wide"><span>SUMMARY</span><div id="summary"></div></div>
<div class="card"><span>TONE</span><div id="tone"></div></div><div class="card"><span>THEMES</span><div id="themes"></div></div>
<div class="card wide"><span>WHY</span><div id="explanation"></div></div></section>
<p class="note">Hosted analysis uses Groq GPT-OSS. The repository's Streamlit edition uses multilingual BERT locally.</p>
</main><script>const $=id=>document.getElementById(id),run=$("run"),status=$("status"),results=$("results");
run.onclick=async()=>{const text=$("text").value.trim();if(!text){status.textContent="Enter some text first.";return}
run.disabled=true;status.className="status";status.textContent="Analyzing…";results.hidden=true;
try{const r=await fetch("/api/analyze",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({text})});
const d=await r.json();if(!r.ok)throw Error(d.error||"Analysis failed");
for(const k of["sentiment","polarity","confidence","summary","tone","explanation"])$(k).textContent=d[k]||"—";
$("themes").textContent=(d.themes||[]).join(" · ");results.hidden=false;status.textContent="Analysis complete."}
catch(e){status.className="status error";status.textContent=e.message}finally{run.disabled=false}};</script></body></html>"""

PROMPT = """Analyze the text. Return only valid JSON with these keys: sentiment
(Very Negative, Negative, Neutral, Positive, or Very Positive), polarity
(negative, neutral, or positive), confidence (a percentage string), summary
(2 concise sentences), themes (array of 2-5 short strings), tone, and explanation
(one concise sentence). Text:\n"""


def _response(start_response, status, body, content_type="application/json"):
    payload = body if isinstance(body, bytes) else body.encode("utf-8")
    start_response(status, [("Content-Type", f"{content_type}; charset=utf-8"),
                            ("Content-Length", str(len(payload))), ("Cache-Control", "no-store")])
    return [payload]


def application(environ, start_response):
    path, method = environ.get("PATH_INFO", "/"), environ.get("REQUEST_METHOD")
    if path == "/" and method == "GET":
        return _response(start_response, "200 OK", HTML, "text/html")
    if path != "/api/analyze" or method != "POST":
        return _response(start_response, "404 Not Found", '{"error":"Not found"}')
    try:
        length = min(int(environ.get("CONTENT_LENGTH") or 0), 50_000)
        text = json.loads(environ["wsgi.input"].read(length)).get("text", "").strip()
        if not text or len(text) > 12_000:
            return _response(start_response, "400 Bad Request", '{"error":"Enter 1–12,000 characters."}')
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return _response(start_response, "503 Service Unavailable",
                             '{"error":"The demo owner still needs to configure GROQ_API_KEY in Vercel."}')
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps({"model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                             "response_format": {"type": "json_object"},
                             "messages": [{"role": "user", "content": PROMPT + text}],
                             "temperature": 0.2}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=45) as upstream:
            result = json.loads(upstream.read())
        content = result["choices"][0]["message"]["content"]
        return _response(start_response, "200 OK", json.dumps(json.loads(content)))
    except (ValueError, KeyError, json.JSONDecodeError):
        return _response(start_response, "502 Bad Gateway",
                         '{"error":"The model returned an invalid response. Try again."}')
    except urllib.error.HTTPError as exc:
        return _response(start_response, "502 Bad Gateway",
                         json.dumps({"error": f"Groq request failed ({exc.code})."}))
    except Exception:
        return _response(start_response, "500 Internal Server Error", '{"error":"Unexpected server error."}')
