from __future__ import annotations

from .models import Action, ActionDecision, DangerLevel


class SafetyManager:
    def evaluate(self, action: Action, safe_mode: bool = True) -> ActionDecision:
        t = action.type.value

        high_risk = {"run_command", "write_file", "move_file", "text_create", "text_str_replace", "text_insert"}
        if t == "run_command":
            cmd = action.args.get("command", "").lower()
            dangerous_patterns = ["rm -rf /", "format ", "del /f /s", ":(){ :|:& };:"]
            if any(p in cmd for p in dangerous_patterns):
                return ActionDecision(
                    danger=DangerLevel.high,
                    reason=f"Hard-blocked dangerous shell command: {cmd}",
                    requires_approval=True
                )
            
        if t in high_risk:
            return ActionDecision(
                danger=DangerLevel.high,
                reason="filesystem/shell mutation",
                requires_approval=True,
            )

        low = {
            "scroll",
            "mouse_move",
            "cursor_position",
            "wait_action",
            "browser_open",
            "browser_screenshot",
            "browser_get_text",
            "browser_accessibility_tree",
            "browser_navigate_back",
            "browser_close",
        }
        medium = {
            "double_click",
            "right_click",
            "middle_click",
            "browser_click",
            "browser_click_coords",
            "browser_type",
            "browser_scroll",
        }
        if t in low:
            return ActionDecision(danger=DangerLevel.low, reason="low risk", requires_approval=False)
        if t == "left_click_drag":
            return ActionDecision(danger=DangerLevel.medium, reason="drag can move data", requires_approval=safe_mode)
        if t in medium:
            return ActionDecision(danger=DangerLevel.medium, reason="medium risk", requires_approval=safe_mode)
        if t == "key_combo":
            keys = action.args.get("keys", "").lower().replace(" ", "")
            dangerous = {"ctrl+alt+del", "win+l", "ctrl+alt+t"}
            if keys in dangerous:
                return ActionDecision(danger=DangerLevel.high, reason="dangerous combo", requires_approval=True)
        return ActionDecision(danger=DangerLevel.low, reason="default", requires_approval=False)
