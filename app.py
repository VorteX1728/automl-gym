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


def build_prompt(
    best_score,
    history,
    user_comment,
    metric,
    remaining_budget,
    remaining_tokens,
    token_budget
):

    tried_models = []
    last_checklist_feedback = []
    last_checklist_summary = {}

    for item in history:
        try:
            model_name = item["action"]["model"]

            if model_name not in tried_models:
                tried_models.append(model_name)

        except Exception:
            pass

    if history:
        try:
            last_observation = history[-1].get("observation", {})
            last_checklist = last_observation.get("checklist", {})
            last_checklist_feedback = last_checklist.get("agent_feedback", [])
            last_checklist_summary = last_checklist.get("summary", {})
        except Exception:
            last_checklist_feedback = []
            last_checklist_summary = {}

    return f"""
You are an AutoML research agent.

Your goal is NOT to repeatedly run the same model.

Your goal is to explore the search space and discover better pipelines.

Current metric:
{metric}

Current best score:
{best_score}

Remaining compute budget:
{remaining_budget}

Remaining token budget:
{remaining_tokens}

Total token budget:
{token_budget}

User comment:
{user_comment}

Models already tested:
{tried_models}

Previous experiments:
{json.dumps(history[-10:], ensure_ascii=False, indent=2)}

LAST CHECKLIST SUMMARY:
{json.dumps(last_checklist_summary, ensure_ascii=False, indent=2)}

LAST ENVIRONMENT FEEDBACK FOR NEXT ACTION:
{json.dumps(last_checklist_feedback, ensure_ascii=False, indent=2)}

CHECKLIST RULES

1.
You must use the environment checklist feedback when choosing the next action.

2.
If the checklist says to try an untested model, choose one of the untested models.

3.
If the checklist reports invalid or sanitized parameters, avoid those parameters next time.

4.
If the checklist reports overfitting, reduce model complexity.

5.
If the checklist reports low token or compute budget, use a smaller and cheaper model.

6.
If the previous action failed, fix the exact error instead of repeating the same action.

AVAILABLE MODELS
    
- xgboost
- lightgbm
- catboost
- random_forest
- hist_gb

IMPORTANT RULES

1.
During the first 5 steps you MUST test different models.

2.
Do NOT repeatedly suggest the same model if another model has not been tested yet.

3.
If xgboost was tested,
consider trying:
- lightgbm
- catboost

4.
If all models were tested,
then start hyperparameter optimization.

5.
Avoid suggesting exactly the same parameters twice.

6.
Prefer exploration before exploitation.

7.
Model diversity is important.

8.
Repeatedly selecting the same model may lead to lower reward.

9.
Try models that were not tested recently.

10.
If a model performed poorly,
try a different model instead of repeating it.

TOKEN BUDGET RULES

1.
Every LLM answer consumes token budget.

2.
Keep responses short. Return only JSON.

3.
When remaining token budget is below 50%, reduce exploration.

4.
When remaining token budget is below 25%, tune the best known model.

5.
When remaining token budget is below 10%, avoid new experiments and choose low-risk parameters.

6.
Do not waste tokens on explanations, markdown, comments, or code blocks.

ACTION FORMAT

{{
    "action": "train",
    "model": "lightgbm",
    "params": {{
        "max_depth": 5,
        "n_estimators": 150,
        "learning_rate": 0.05
    }}
}}

Return ONLY valid JSON.

Do not write explanations.
Do not write markdown.
Do not write code blocks.
Return ONLY JSON.
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
                max(0, env.token_budget - env.total_token_cost),
                env.token_budget
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