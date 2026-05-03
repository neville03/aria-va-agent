import json, os
from datetime import datetime
from pathlib import Path
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config ─────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"
STATE_FILE = Path("state.json")

# ── State ──────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tasks": [], "next_id": 1}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ── Tools ──────────────────────────────────────────
def add_task(title):
    state = load_state()
    task = {
        "id": state["next_id"],
        "title": title,
        "done": False,
        "created": datetime.now().isoformat()
    }
    state["tasks"].append(task)
    state["next_id"] += 1
    save_state(state)
    return {"task": task}

def list_tasks():
    return load_state()

TOOL_FUNCTIONS = {
    "add_task": add_task,
    "list_tasks": list_tasks,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

SYSTEM_PROMPT = "You are ARIA, a helpful assistant."

# ── Agent ──────────────────────────────────────────
def run_agent(messages):
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content

    for tc in msg.tool_calls:
        fn = TOOL_FUNCTIONS[tc.function.name]
        args = json.loads(tc.function.arguments)
        result = fn(**args)

        messages.append({
            "role": "tool",
            "content": json.dumps(result)
        })

    return "Done."

# ── Flask ──────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "running", "routes": ["/chat", "/state"]})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    reply = run_agent(messages)
    return jsonify({"reply": reply})

@app.route("/state")
def state():
    return jsonify(load_state())

if __name__ == "__main__":
    app.run(port=5050)
