"""
StrikeForce VA Agent — Groq-powered personal assistant agent.

Architecture:
  - Tool definitions  : what the agent CAN do (JSON Schema)
  - Tool execution    : the actual Python functions
  - Agent loop        : reason → tool call → observe → repeat → answer
  - Flask API         : exposes /chat and /state endpoints to the frontend
"""

import json, os, re
from datetime import datetime
from pathlib import Path
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"
STATE_FILE   = Path(__file__).parent / "state.json"

# ── State store (simple JSON file) ────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tasks": [], "reminders": [], "next_id": 1}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ── Tool implementations ───────────────────────────────────────────────────────

def add_task(title: str, priority: str = "normal", due_date: str = None) -> dict:
    state = load_state()
    task = {
        "id": state["next_id"],
        "title": title,
        "priority": priority,
        "due_date": due_date,
        "done": False,
        "created": datetime.now().isoformat()
    }
    state["tasks"].append(task)
    state["next_id"] += 1
    save_state(state)
    return {"success": True, "task": task, "message": f"Task #{task['id']} added: '{title}'"}

def list_tasks(filter: str = "pending") -> dict:
    state = load_state()
    tasks = state["tasks"]
    if filter == "pending":
        tasks = [t for t in tasks if not t["done"]]
    elif filter == "done":
        tasks = [t for t in tasks if t["done"]]
    return {"tasks": tasks, "count": len(tasks), "filter": filter}

def complete_task(task_id: int) -> dict:
    state = load_state()
    for t in state["tasks"]:
        if t["id"] == task_id:
            t["done"] = True
            save_state(state)
            return {"success": True, "message": f"Task #{task_id} marked as done."}
    return {"success": False, "message": f"Task #{task_id} not found."}

def set_reminder(title: str, datetime_str: str, note: str = "") -> dict:
    state = load_state()
    reminder = {
        "id": state["next_id"],
        "title": title,
        "datetime": datetime_str,
        "note": note,
        "created": datetime.now().isoformat()
    }
    state["reminders"].append(reminder)
    state["next_id"] += 1
    save_state(state)
    return {"success": True, "reminder": reminder, "message": f"Reminder set: '{title}' at {datetime_str}"}

def list_reminders() -> dict:
    state = load_state()
    return {"reminders": state["reminders"], "count": len(state["reminders"])}

def cancel_reminder(reminder_id: int) -> dict:
    state = load_state()
    before = len(state["reminders"])
    state["reminders"] = [r for r in state["reminders"] if r["id"] != reminder_id]
    save_state(state)
    removed = before - len(state["reminders"])
    return {"success": removed > 0, "message": f"Reminder #{reminder_id} {'cancelled' if removed else 'not found'}."}

def draft_email(to: str, subject: str, context: str, tone: str = "professional") -> dict:
    # This tool returns a draft — the LLM will write the email body in its final answer
    return {
        "instruction": "draft_email",
        "to": to,
        "subject": subject,
        "context": context,
        "tone": tone,
        "note": "Write a complete email draft based on these parameters in your final response."
    }

def draft_proposal(client_name: str, service: str, context: str) -> dict:
    return {
        "instruction": "draft_proposal",
        "client_name": client_name,
        "service": service,
        "context": context,
        "note": "Write a structured business proposal based on these parameters in your final response."
    }

def get_daily_briefing() -> dict:
    state = load_state()
    pending = [t for t in state["tasks"] if not t["done"]]
    reminders = state["reminders"]
    today = datetime.now().strftime("%A, %d %B %Y")
    return {
        "date": today,
        "pending_tasks": pending,
        "pending_task_count": len(pending),
        "reminders": reminders,
        "reminder_count": len(reminders),
        "note": "Summarize this as a friendly morning briefing in your final response."
    }

# ── Tool registry ──────────────────────────────────────────────────────────────
TOOL_FUNCTIONS = {
    "add_task": add_task,
    "list_tasks": list_tasks,
    "complete_task": complete_task,
    "set_reminder": set_reminder,
    "list_reminders": list_reminders,
    "cancel_reminder": cancel_reminder,
    "draft_email": draft_email,
    "draft_proposal": draft_proposal,
    "get_daily_briefing": get_daily_briefing,
}

# ── Tool definitions (JSON Schema for the LLM) ─────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a new task to the user's task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":     {"type": "string",  "description": "Task description"},
                    "priority":  {"type": "string",  "enum": ["low","normal","high"], "description": "Task priority"},
                    "due_date":  {"type": "string",  "description": "Due date as a readable string e.g. 'Friday 2 May'"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List the user's tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "enum": ["all","pending","done"], "description": "Which tasks to show"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a task as complete by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "The ID of the task to complete"}
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":        {"type": "string", "description": "What to remind about"},
                    "datetime_str": {"type": "string", "description": "When e.g. 'Tomorrow at 9am', 'Friday 3pm'"},
                    "note":         {"type": "string", "description": "Optional additional context"}
                },
                "required": ["title", "datetime_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all of the user's active reminders.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancel a reminder by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "integer", "description": "The ID of the reminder to cancel"}
                },
                "required": ["reminder_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Draft a professional email. Provide recipient, subject and context — the agent will write the full body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient name or email"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "context": {"type": "string", "description": "What the email is about / key points to include"},
                    "tone":    {"type": "string", "enum": ["professional","friendly","formal","urgent"], "description": "Tone of the email"}
                },
                "required": ["to", "subject", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_proposal",
            "description": "Draft a business proposal for a client.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string", "description": "Name of the client or company"},
                    "service":     {"type": "string", "description": "Service or product being proposed"},
                    "context":     {"type": "string", "description": "Key details, scope, pricing hints, goals"}
                },
                "required": ["client_name", "service", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_briefing",
            "description": "Get a summary of today's tasks and reminders as a morning briefing.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ARIA — an elite personal assistant AI built for busy professionals and business owners.

Your personality: calm, sharp, proactive. You speak like a world-class EA — concise, confident, and always one step ahead.

Your capabilities:
- Task management (add, list, complete tasks)
- Reminders (set, list, cancel)
- Draft professional emails with complete, ready-to-send bodies
- Draft business proposals — structured, persuasive, client-ready
- Give daily briefings summarising what's on the user's plate

How you work:
1. Understand the user's request
2. Use the right tool(s) to take action or gather data
3. Give a clear, concise final response — never just echo back tool output

For drafting tasks (emails, proposals): always call the draft tool first to register intent, then write the FULL draft in your response. Never leave the user with an incomplete draft.

For briefings: call get_daily_briefing, then write a warm, structured morning summary.

Today's date: """ + datetime.now().strftime("%A, %d %B %Y")

# ── Agent loop ─────────────────────────────────────────────────────────────────
def run_agent(messages: list, api_key: str) -> tuple[str, list]:
    """
    Core ReAct loop:
    1. Send messages + tools to Groq
    2. If model calls a tool → execute it, append result, loop
    3. If model gives text response → return it
    """
    client = Groq(api_key=api_key)
    steps = []   # for frontend transparency
    MAX_ITER = 8

    for _ in range(MAX_ITER):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=2048,
            temperature=0.4,
        )
        msg = response.choices[0].message

        # If no tool calls → final answer
        if not msg.tool_calls:
            return msg.content, steps

        # Process tool calls
        tool_results = []
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            # Execute the tool
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            steps.append({
                "tool": fn_name,
                "args": fn_args,
                "result": result
            })

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

        # Append assistant message (with tool_calls) and all tool results
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})
        for tr in tool_results:
            messages.append(tr)

    return "I've reached my thinking limit. Please try a simpler request.", steps

# ── Flask API ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# In-memory conversation history per session (keyed by session_id)
conversations = {}

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.json or {}
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    api_key    = data.get("api_key", GROQ_API_KEY)

    if not user_msg:
        return jsonify({"error": "No message provided"}), 400
    if not api_key:
        return jsonify({"error": "No Groq API key provided"}), 400

    # Build or continue conversation
    if session_id not in conversations:
        conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversations[session_id].append({"role": "user", "content": user_msg})

    try:
        answer, steps = run_agent(list(conversations[session_id]), api_key)
        # Append final assistant answer to history
        conversations[session_id].append({"role": "assistant", "content": answer})
        return jsonify({"reply": answer, "steps": steps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/state", methods=["GET"])
def state():
    return jsonify(load_state())

@app.route("/clear", methods=["POST"])
def clear_conversation():
    data       = request.json or {}
    session_id = data.get("session_id", "default")
    if session_id in conversations:
        del conversations[session_id]
    return jsonify({"success": True})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL, "tools": list(TOOL_FUNCTIONS.keys())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
