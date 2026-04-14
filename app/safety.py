from __future__ import annotations

from pathlib import Path

from .models import Action, ActionDecision, DangerLevel, SafetyConfig


class SafetyManager:
    def __init__(self, workspace: Path, config: SafetyConfig):
        self.workspace = workspace.resolve()
        self.config = config

    def update(self, config: SafetyConfig) -> SafetyConfig:
        self.config = config
        return self.config

    def evaluate_action(self, action: Action) -> ActionDecision:
        if action.type.value == "run_command":
            return self._evaluate_command(action)
        if action.type.value in {"mouse_click", "keyboard_type"}:
            danger = DangerLevel.medium if self.config.safe_mode else DangerLevel.low
            return ActionDecision(action_id=action.id, allowed=True, danger=danger, reason="GUI action allowed with review")
        return ActionDecision(action_id=action.id, allowed=True, danger=DangerLevel.low, reason="Allowed by safety policy")

    def _evaluate_command(self, action: Action) -> ActionDecision:
        command = str(action.args.get("command", "")).strip().lower()
        if not command:
            return ActionDecision(action_id=action.id, allowed=False, danger=DangerLevel.blocked, reason="Empty command")

        for forbidden in self.config.command_denylist:
            if forbidden.lower() in command:
                return ActionDecision(
                    action_id=action.id,
                    allowed=False,
                    danger=DangerLevel.blocked,
                    reason=f"Command matched denylist rule: {forbidden}",
                )

        if self.config.safe_mode:
            lead = command.split()[0]
            if self.config.command_allowlist and lead not in self.config.command_allowlist:
                return ActionDecision(
                    action_id=action.id,
                    allowed=False,
                    danger=DangerLevel.blocked,
                    reason=f"Safe mode blocks command '{lead}' (not in allowlist)",
                )

        danger = DangerLevel.medium if any(x in command for x in ["sudo", "apt", "pip install", "docker", "curl | sh"]) else DangerLevel.low
        return ActionDecision(action_id=action.id, allowed=True, danger=danger, reason="Command allowed")
