import os
import json
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

from env.automl_env import AutoMLEnv
from agents.llm_agent import LLMAgent


UPLOAD_FOLDER = "uploads"
SUBMISSION_FOLDER = "submissions"
STATE_FILE = "state.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBMISSION_FOLDER, exist_ok=True)

app = Flask(__name__)

RUN_LOCK = threading.Lock()
RUNNING = False


def set_running(value):
    global RUNNING

    with RUN_LOCK:
        RUNNING = value


def is_running():
    with RUN_LOCK:
        return RUNNING


def save_state(data):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "status": "idle",
            "progress": 0,
            "logs": "",
            "best_result": {},
            "submission_file": None,
            "is_running": False
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["is_running"] = is_running()

    return data


def compact_history(history, max_items=5):
    compact = []

    for item in history[-max_items:]:
        action = item.get("action", {})
        observation = item.get("observation", {})

        compact.append({
            "model": action.get("model"),
            "params": action.get("params", {}),
            "success": observation.get("success"),
            "val_score": observation.get("val_score"),
            "objective_value": observation.get("objective_value"),
            "reward": observation.get("reward"),
            "overfit_gap": observation.get("overfit_gap"),
            "cv_std": observation.get("cv_std"),
            "error": observation.get("error"),
            "checklist_feedback": (
                observation
                .get("checklist", {})
                .get("agent_feedback", [])[:3]
            )
        })

    return compact


def build_prompt(
    best_score,
    history,
    user_comment,
    metric,
    remaining_budget,
    remaining_tokens
):
    short_history = compact_history(
        history,
        max_items=4
    )

    return f"""
You are an AutoML agent for tabular data.

Metric:
{metric}

Current best objective score:
{best_score}

Remaining compute budget:
{remaining_budget}

Remaining token budget:
{remaining_tokens}

User comment:
{user_comment}

Recent compact experiments:
{json.dumps(short_history, ensure_ascii=False, indent=2)}

Allowed models:
- xgboost
- lightgbm
- catboost
- random_forest
- hist_gb

Rules:
1. Return ONLY JSON.
2. Keep response short.
3. Do not explain.
4. Use different models during early exploration.
5. Avoid repeating same model and same params.
6. If remaining token budget is low, tune the current best model.
7. Do not use markdown.
8. Do not use code blocks.
9. Return JSON under 80 tokens.

JSON format:

{{
    "action": "train",
    "model": "lightgbm",
    "params": {{
        "max_depth": 5,
        "n_estimators": 120,
        "learning_rate": 0.05
    }}
}}
"""
def run_automl(
    train_path,
    test_path,
    target,
    user_comment,
    steps,
    llm_model,
    metric
):
    logs = ""

    try:
        set_running(True)

        save_state({
            "status": "Initializing",
            "progress": 0,
            "logs": "",
            "best_result": {},
            "submission_file": None,
            "is_running": True
        })

        env = AutoMLEnv(
            train_path,
            target_column=target,
            metric=metric
        )

        agent = LLMAgent(
            model_name=llm_model
        )

        best_objective = -float("inf")
        best_result = {}

        for step in range(steps):
            progress = int((step / max(steps, 1)) * 90)

            logs += "\n====================\n"
            logs += f"STEP {step}\n"
            logs += "====================\n"

            save_state({
                "status": f"Step {step}: generating action",
                "progress": progress,
                "logs": logs,
                "best_result": best_result,
                "submission_file": None,
                "is_running": True
            })

            prompt = build_prompt(
                best_objective,
                env.history,
                user_comment,
                metric,
                max(0, env.compute_budget - env.total_compute_cost),
                max(0, env.token_budget - env.total_token_cost)
            )

            action = agent.act(prompt)

            logs += "\nACTION:\n"
            logs += json.dumps(
                action,
                ensure_ascii=False,
                indent=2
            )
            logs += "\n"

            save_state({
                "status": f"Step {step}: training model",
                "progress": min(progress + 5, 95),
                "logs": logs,
                "best_result": best_result,
                "submission_file": None,
                "is_running": True
            })

            result = env.step(action)

            logs += "\nRESULT:\n"
            logs += json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
            logs += "\n"

            if result.get("budget_exhausted"):
                logs += "\nBudget exhausted. Stopping.\n"
                break

            if result.get("success"):
                if result["objective_value"] > best_objective:
                    best_objective = result["objective_value"]
                    best_result = result
            else:
                logs += "\nMODEL FAILED:\n"
                logs += result.get("error", "")
                logs += "\n"

            save_state({
                "status": f"Step {step}: completed",
                "progress": int(((step + 1) / max(steps, 1)) * 90),
                "logs": logs,
                "best_result": best_result,
                "submission_file": None,
                "is_running": True
            })

        logs += "\nCreating submission...\n"

        save_state({
            "status": "Creating submission",
            "progress": 95,
            "logs": logs,
            "best_result": best_result,
            "submission_file": None,
            "is_running": True
        })

        submission = env.predict_submission(
            test_path=test_path
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        submission_filename = (
            f"submission_{timestamp}.csv"
        )

        submission_path = os.path.join(
            SUBMISSION_FOLDER,
            submission_filename
        )

        submission.to_csv(
            submission_path,
            index=False
        )

        logs += f"\nSubmission saved: {submission_path}\n"
        logs += f"Rows in submission: {len(submission)}\n"

        save_state({
            "status": "FINISHED",
            "progress": 100,
            "logs": logs,
            "best_result": best_result,
            "submission_file": submission_filename,
            "is_running": False
        })

    except Exception as error:
        logs += "\nFATAL ERROR:\n"
        logs += str(error)

        save_state({
            "status": "ERROR",
            "progress": 100,
            "logs": logs,
            "best_result": {},
            "submission_file": None,
            "is_running": False
        })

    finally:
        set_running(False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    if is_running():
        return jsonify({
            "status": "already_running",
            "message": "AutoML process is already running"
        }), 409

    train_file = request.files["file"]
    test_file = request.files.get("test_file")

    target = request.form["target"]
    comment = request.form.get("comment", "")
    steps = int(request.form.get("steps", 5))
    llm_model = request.form["llm_model"]
    metric = request.form["metric"]

    train_path = os.path.join(
        UPLOAD_FOLDER,
        train_file.filename
    )

    train_file.save(train_path)

    test_path = None

    if test_file and test_file.filename:
        test_path = os.path.join(
            UPLOAD_FOLDER,
            test_file.filename
        )

        test_file.save(test_path)

    set_running(True)

    thread = threading.Thread(
        target=run_automl,
        args=(
            train_path,
            test_path,
            target,
            comment,
            steps,
            llm_model,
            metric
        ),
        daemon=True
    )

    thread.start()

    return jsonify({
        "status": "started"
    })


@app.route("/state")
def state():
    return jsonify(load_state())


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(
        SUBMISSION_FOLDER,
        filename,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        threaded=True
    )
