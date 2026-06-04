import json
import re
import ollama


class LLMAgent:
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def _extract_json(self, text: str):
        if "</think>" in text:
            text = text.split("</think>")[-1]

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return None

    def act(self, prompt):
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response["message"]["content"]

        prompt_tokens = response.get("prompt_eval_count", self._estimate_tokens(prompt))
        response_tokens = response.get("eval_count", self._estimate_tokens(content))
        total_tokens = prompt_tokens + response_tokens

        json_text = self._extract_json(content)

        try:
            parsed = json.loads(json_text) if json_text else {}
        except Exception:
            parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        parsed.setdefault("action", "train")
        parsed.setdefault("model", "xgboost")
        parsed.setdefault("params", {})

        parsed["_token_info"] = {
            "prompt_tokens": int(prompt_tokens),
            "response_tokens": int(response_tokens),
            "total_tokens": int(total_tokens)
        }

        return parsed
