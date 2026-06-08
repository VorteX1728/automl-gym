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
            "run_stats": {},
            "submission_file": None,
            "model_state": "IDLE",
            "llm_comparison": [],
            "is_running": False
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["is_running"] = is_running()

    if "run_stats" not in data:
        data["run_stats"] = {}

    if "llm_comparison" not in data:
        data["llm_comparison"] = []

    if "model_state" not in data:
        data["model_state"] = "IDLE"

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
    remaining_tokens,
    agent_name=None,
    compare_mode=False
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

Agent name:
{agent_name or "single_agent"}

Comparison mode:
{compare_mode}

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
10. In comparison mode, optimize independently from other LLM agents.

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


def make_run_stats(result, env, step):
    return {
        "total_compute_cost": result.get("total_compute_cost"),
        "compute_cost": result.get("compute_cost"),
        "remaining_budget": result.get("remaining_budget"),
        "total_token_cost": result.get("total_token_cost"),
        "remaining_tokens": result.get("remaining_tokens"),
        "token_budget": result.get("token_budget"),
        "current_step": step,
        "total_candidates": len(env.candidates)
    }


def parse_llm_models(value):
    if not value:
        return ["deepseek-r1:7b"]

    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).replace("\n", ",").split(",")

    models = []

    for item in raw:
        model = str(item).strip()

        if model and model not in models:
            models.append(model)

    return models or ["deepseek-r1:7b"]


def empty_agent_summary(model_name):
    return {
        "llm_model": model_name,
        "status": "pending",
        "best_model": None,
        "best_score": None,
        "best_reward": None,
        "best_candidate_id": None,
        "successful_steps": 0,
        "failed_steps": 0,
        "total_token_cost": 0,
        "total_compute_cost": 0.0,
        "remaining_tokens": None,
        "remaining_budget": None,
        "submission_file": None,
        "error": None
    }


def update_agent_summary(summary, result, env):
    summary["total_token_cost"] = int(getattr(env, "total_token_cost", 0))
    summary["total_compute_cost"] = float(getattr(env, "total_compute_cost", 0.0))
    summary["remaining_tokens"] = int(
        max(0, getattr(env, "token_budget", 0) - getattr(env, "total_token_cost", 0))
    )
    summary["remaining_budget"] = float(
        max(0, getattr(env, "compute_budget", 0) - getattr(env, "total_compute_cost", 0))
    )

    if result and result.get("success"):
        summary["successful_steps"] += 1

        score = result.get("selection_score", result.get("reward"))

        if (
            summary["best_reward"] is None
            or score > summary["best_reward"]
        ):
            summary["best_reward"] = score
            summary["best_score"] = result.get("val_score")
            summary["best_model"] = result.get("best_model") or result.get("model")
            summary["best_candidate_id"] = result.get("candidate_id")

    elif result:
        summary["failed_steps"] += 1
        summary["error"] = result.get("error")

    return summary


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

        llm_models = parse_llm_models(llm_model)
        compare_mode = len(llm_models) > 1

        llm_comparison = [
            empty_agent_summary(model_name)
            for model_name in llm_models
        ]

        save_state({
            "status": "Initializing",
            "progress": 0,
            "logs": "",
            "best_result": {},
            "run_stats": {},
            "submission_file": None,
            "model_state": "EDA",
            "llm_comparison": llm_comparison,
            "is_running": True
        })

        global_best_result = {}
        global_best_selection_score = -float("inf")
        global_submission_file = None
        global_run_stats = {}

        for agent_index, current_llm_model in enumerate(llm_models):

            llm_comparison[agent_index]["status"] = "running"

            logs += "\n====================\n"
            logs += f"LLM AGENT {agent_index + 1}/{len(llm_models)}: {current_llm_model}\n"
            logs += "====================\n"

            save_state({
                "status": f"Agent {agent_index + 1}/{len(llm_models)}: initializing",
                "model_state": "EDA",
                "progress": int((agent_index / max(len(llm_models), 1)) * 95),
                "logs": logs,
                "best_result": global_best_result,
                "run_stats": global_run_stats,
                "submission_file": global_submission_file,
                "llm_comparison": llm_comparison,
                "is_running": True
            })

            env = AutoMLEnv(
                train_path,
                target_column=target,
                metric=metric
            )

            agent = LLMAgent(
                model_name=current_llm_model
            )

            best_selection_score = -float("inf")
            best_result = {}
            run_stats = {}

            for step in range(steps):
                progress = int(
                    (
                        (agent_index + (step / max(steps, 1)))
                        / max(len(llm_models), 1)
                    ) * 95
                )

                logs += "\n--------------------\n"
                logs += f"AGENT: {current_llm_model}\n"
                logs += f"STEP {step}\n"
                logs += "--------------------\n"

                save_state({
                    "status": f"{current_llm_model}: step {step} generating action",
                    "model_state": "MODEL_SELECTION",
                    "progress": progress,
                    "logs": logs,
                    "best_result": global_best_result or best_result,
                    "run_stats": run_stats,
                    "submission_file": global_submission_file,
                    "llm_comparison": llm_comparison,
                    "is_running": True
                })

                prompt = build_prompt(
                    best_selection_score,
                    env.history,
                    user_comment,
                    metric,
                    max(0, env.compute_budget - env.total_compute_cost),
                    max(0, env.token_budget - env.total_token_cost),
                    agent_name=current_llm_model,
                    compare_mode=compare_mode
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
                    "status": f"{current_llm_model}: step {step} training",
                    "model_state": "TRAINING",
                    "progress": min(progress + 3, 98),
                    "logs": logs,
                    "best_result": global_best_result or best_result,
                    "run_stats": run_stats,
                    "submission_file": global_submission_file,
                    "llm_comparison": llm_comparison,
                    "is_running": True
                })

                result = env.step(action)

                run_stats = make_run_stats(
                    result=result,
                    env=env,
                    step=step
                )

                logs += "\nRESULT:\n"
                logs += json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                )
                logs += "\n"

                llm_comparison[agent_index] = update_agent_summary(
                    llm_comparison[agent_index],
                    result,
                    env
                )

                if result.get("budget_exhausted"):
                    logs += "\nBudget exhausted for this agent. Stopping agent run.\n"
                    break

                if result.get("success"):

                    selection_score = result.get(
                        "selection_score",
                        result.get("reward", -float("inf"))
                    )

                    if selection_score > best_selection_score:
                        best_selection_score = selection_score
                        best_result = result

                    if selection_score > global_best_selection_score:
                        global_best_selection_score = selection_score
                        global_best_result = result
                        global_best_result["llm_model"] = current_llm_model
                        global_run_stats = run_stats

                else:
                    logs += "\nMODEL FAILED:\n"
                    logs += result.get("error", "")
                    logs += "\n"

                save_state({
                    "status": f"{current_llm_model}: step {step} completed",
                    "model_state": "EVALUATION",
                    "progress": int(
                        (
                            (agent_index + ((step + 1) / max(steps, 1)))
                            / max(len(llm_models), 1)
                        ) * 95
                    ),
                    "logs": logs,
                    "best_result": global_best_result or best_result,
                    "run_stats": global_run_stats or run_stats,
                    "submission_file": global_submission_file,
                    "llm_comparison": llm_comparison,
                    "is_running": True
                })

            if best_result:
                logs += f"\nCreating submission for agent {current_llm_model}...\n"

                try:
                    submission = env.predict_submission(
                        test_path=test_path
                    )

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    safe_agent_name = (
                        current_llm_model
                        .replace(":", "_")
                        .replace("/", "_")
                        .replace("\\", "_")
                    )

                    submission_filename = (
                        f"submission_{safe_agent_name}_{timestamp}.csv"
                    )

                    submission_path = os.path.join(
                        SUBMISSION_FOLDER,
                        submission_filename
                    )

                    submission.to_csv(
                        submission_path,
                        index=False
                    )

                    llm_comparison[agent_index]["submission_file"] = submission_filename

                    logs += f"Submission saved: {submission_path}\n"
                    logs += f"Rows in submission: {len(submission)}\n"

                    agent_best_reward = llm_comparison[agent_index].get("best_reward")

                    global_best_reward = None
                    if global_best_result:
                        global_best_reward = global_best_result.get(
                            "selection_score",
                            global_best_result.get("reward")
                        )

                    if (
                        agent_best_reward is not None
                        and global_best_reward is not None
                        and agent_best_reward >= global_best_reward
                    ):
                        global_submission_file = submission_filename

                    if (
                        global_best_result
                        and global_best_result.get("llm_model") == current_llm_model
                    ):
                        global_submission_file = submission_filename

                except Exception as error:
                    logs += "\nSubmission creation failed for agent:\n"
                    logs += str(error)
                    logs += "\n"
                    llm_comparison[agent_index]["error"] = str(error)

            if (
                global_submission_file
                and global_best_result
                and global_best_result.get("llm_model") == current_llm_model
            ):
                llm_comparison[agent_index]["submission_file"] = global_submission_file

            llm_comparison[agent_index]["status"] = "finished"

            save_state({
                "status": f"Agent {agent_index + 1}/{len(llm_models)} finished",
                "model_state": "EVALUATION",
                "progress": int(((agent_index + 1) / max(len(llm_models), 1)) * 95),
                "logs": logs,
                "best_result": global_best_result,
                "run_stats": global_run_stats,
                "submission_file": global_submission_file,
                "llm_comparison": llm_comparison,
                "is_running": True
            })

        save_state({
            "status": "FINISHED",
            "model_state": "FINISHED",
            "progress": 100,
            "logs": logs,
            "best_result": global_best_result,
            "run_stats": global_run_stats,
            "submission_file": global_submission_file,
            "llm_comparison": llm_comparison,
            "is_running": False
        })

    except Exception as error:
        logs += "\nFATAL ERROR:\n"
        logs += str(error)

        save_state({
            "status": "ERROR",
            "model_state": "ERROR",
            "progress": 100,
            "logs": logs,
            "best_result": {},
            "run_stats": {},
            "submission_file": None,
            "llm_comparison": [],
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
    llm_models = request.form.getlist("llm_model")

    if not llm_models:
        llm_models = [request.form.get("llm_model", "deepseek-r1:7b")]

    llm_model = ",".join(llm_models)

    metric = request.form["metric"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    safe_train_name = train_file.filename.replace("\\", "_").replace("/", "_")

    train_path = os.path.join(
        UPLOAD_FOLDER,
        f"{timestamp}_train_{safe_train_name}"
    )

    train_file.save(train_path)

    test_path = None

    if test_file and test_file.filename:

        safe_test_name = test_file.filename.replace("\\", "_").replace("/", "_")

        test_path = os.path.join(
            UPLOAD_FOLDER,
            f"{timestamp}_test_{safe_test_name}"
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
