import pytest

from app.models import ToolError
from app.text_editor import TextEditorTool


def test_text_editor(workspace):
    t = TextEditorTool(workspace)
    (workspace / "d1").mkdir()
    (workspace / "d1" / "f.txt").write_text("a\nb\nc\nd\n")
    assert "f.txt" in t.view("d1").output
    full = t.view("d1/f.txt").output
    assert "1:" in full and "4:" in full
    ranged = t.view("d1/f.txt", [2, 4]).output
    assert "1:" not in ranged and "2:" in ranged and "4:" in ranged

    assert t.str_replace("d1/f.txt", "b", "B").ok
    with pytest.raises(ToolError):
        t.str_replace("d1/f.txt", "zz", "x")
    (workspace / "d1" / "m.txt").write_text("x\nx\n")
    with pytest.raises(ToolError):
        t.str_replace("d1/m.txt", "x", "y")

    (workspace / "d1" / "i.txt").write_text("1\n2\n")
    t.insert("d1/i.txt", 0, "0")
    assert (workspace / "d1" / "i.txt").read_text().startswith("0")
    t.insert("d1/i.txt", 2, "X")
    assert "X" in (workspace / "d1" / "i.txt").read_text().splitlines()[2]
    t.undo_edit("d1/i.txt")
    with pytest.raises(ToolError):
        t.undo_edit("d1/none.txt")
    with pytest.raises(ToolError):
        t.view("../escape.txt")
