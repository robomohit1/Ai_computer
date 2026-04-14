from __future__ import annotations

import json
from typing import Callable

import httpx

from .models import Action, ActionType, EvaluateResponse, ProviderConfig, ProviderPlanResponse


SYSTEM_PROMPT = """You are a planning engine for a local autonomous computer agent.
Return ONLY valid JSON with shape: {\"reasoning\": str, \"actions\": [{\"id\": str, \"type\": str, \"args\": object, \"explanation\": str, \"requires_approval\": bool}]}
Use only action types: finish, run_command, read_file, write_file, move_file, mouse_click, keyboard_type, screenshot, ocr_image, api_call.
Never output markdown. Never output prose outside JSON.
"""

EVALUATE_PROMPT = """You are an execution evaluator.
Return ONLY JSON with shape: {\"done\": bool, \"summary\": str, \"next_prompt\": str}.
- done=true if goal is complete.
- done=false if more actions are needed; set next_prompt to the next planning instruction.
"""


class ProviderError(RuntimeError):
    pass


class PlannerProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config

    async def plan(self, provider: str, model: str, prompt: str, context: str = "") -> ProviderPlanResponse:
        provider = provider.lower()
        decorated_prompt = prompt if not context else f"Context:\n{context}\n\nGoal:\n{prompt}"
        if provider == "openai":
            if not self.config.openai_api_key:
                return self._fallback_plan(decorated_prompt)
            raw = await self._chat_openai(model, SYSTEM_PROMPT, decorated_prompt)
        elif provider == "anthropic":
            if not self.config.anthropic_api_key:
                return self._fallback_plan(decorated_prompt)
            raw = await self._chat_anthropic(model, SYSTEM_PROMPT, decorated_prompt)
        else:
            raise ProviderError(f"Unsupported provider: {provider}")

        parsed = self._parse_json_with_retries(lambda fix: self._fix_json(provider, model, raw, fix), raw)
        return ProviderPlanResponse(**parsed)

    async def evaluate(self, provider: str, model: str, goal: str, execution_log: str) -> EvaluateResponse:
        eval_input = f"Goal:\n{goal}\n\nExecution log:\n{execution_log[-7000:]}"
        provider = provider.lower()
        if provider == "openai" and self.config.openai_api_key:
            raw = await self._chat_openai(model, EVALUATE_PROMPT, eval_input)
        elif provider == "anthropic" and self.config.anthropic_api_key:
            raw = await self._chat_anthropic(model, EVALUATE_PROMPT, eval_input)
        else:
            return EvaluateResponse(done=True, summary="Fallback evaluator: one pass completed.")

        parsed = self._parse_json_with_retries(lambda fix: self._fix_json(provider, model, raw, fix), raw)
        return EvaluateResponse(**parsed)

    def _parse_json_with_retries(self, fixer: Callable[[str], str], initial_text: str, max_attempts: int = 3) -> dict:
        text = initial_text
        for attempt in range(max_attempts):
            try:
                return json.loads(text)
            except Exception:
                if attempt == max_attempts - 1:
                    raise ProviderError(f"Model returned invalid JSON after retries: {initial_text}")
                text = fixer("Return corrected JSON only. No markdown.")
        raise ProviderError("Unreachable parser state")

    def _fix_json(self, provider: str, model: str, raw: str, instruction: str) -> str:
        """Fallback local fix path; provider-independent best effort."""
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return raw[start:end]
        except ValueError:
            raise ProviderError(f"Unable to recover JSON: {instruction}")

    async def _chat_openai(self, model: str, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.config.openai_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _chat_anthropic(self, model: str, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.config.anthropic_base_url.rstrip('/')}/messages"
        payload = {
            "model": model,
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            chunks = [part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"]
            return "".join(chunks)

    def _fallback_plan(self, prompt: str) -> ProviderPlanResponse:
        return ProviderPlanResponse(
            reasoning="No provider key configured; writing request to file for manual handling.",
            actions=[
                Action(
                    id="a1",
                    type=ActionType.write_file,
                    args={
                        "path": "agent_output/plan.txt",
                        "content": (
                            "No provider API key configured.\n"
                            "Prompt captured below for manual execution:\n\n"
                            f"{prompt}\n"
                        ),
                    },
                    explanation="Persist prompt for manual review when no model key is configured.",
                    requires_approval=True,
                ),
                Action(
                    id="a2",
                    type=ActionType.finish,
                    args={"message": "Fallback plan completed"},
                    explanation="End fallback task cleanly",
                    requires_approval=False,
                ),
            ],
        )
