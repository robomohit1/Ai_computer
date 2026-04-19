# AI Computer 🚀

An autonomous AI agent that controls your computer using natural language. 

AI Computer is a high-performance, open-source desktop application that acts as a bridge between your ideas and your machine. It is designed to be **extensible**, **provider-agnostic**, and **production-ready**, supporting multiple LLM backends (Claude, Gemini, GPT-4) and specialized operating modes.

---

## 📂 Project Structure

| Path | Description |
| :--- | :--- |
| `app/` | **Backend Core** (FastAPI/Python) |
| `app/agent.py` | The "Brain" - Asynchronous loop for planning and task execution. |
| `app/providers.py` | LLM Orchestration & Mode Detection logic. |
| `app/models.py` | Data schemas for actions, sub-tasks, and results. |
| `app/tools.py` | Physical execution layer (Shell, Filesystem, PyAutoGUI, etc.). |
| `app/safety.py` | Risk evaluation & Hard-blocked command lists. |
| `app/permissions.py` | Granular security scope enforcement (filesystem vs. system). |
| `app/plugins/` | Extensible components like `browser_plugin.py` (Playwright). |
| `static/` | **Frontend UI** (HTML5, Vanilla JS, CSS3). |
| `desktop/` | **Native Wrapper** (Electron shell for Windows/Mac/Linux). |
| `workspace/` | Default directory for agent-created projects. |

---

## 💡 Operating Modes

AI Computer automatically detects the best mode for your task, or lets you choose manually:

### 1. 💻 Coding Mode (Files & Shell)
- **Use Case**: Building apps, refactoring code, running scripts.
- **How it works**: Uses zero-vision overhead. It interacts purely via shell commands and text-editor tools. 
- **Benefits**: Extremely fast and accurate for technical tasks.

### 2. 🌐 Computer Use (Browser)
- **Use Case**: Searching the web, filling forms, booking flights, web scraping.
- **How it works**: Spins up an isolated, headless **Playwright** browser.
- **Benefits**: Runs completely in the **background**. You can keep using your computer while the agent browses the web on a separate virtual instance.

### 3. 🖥️ Computer (Native Desktop)
- **Use Case**: Controlling native apps (Discord, Notepad, Settings, Copilot App).
- **How it works**: Uses Computer Vision (screenshots) and **PyAutoGUI** to move your physical mouse and keyboard.
- **Safety Protocol**: Includes a **"Take a Break" Overlay**. When active, the UI blocks inputs and warns: *AI is driving*. 
- **Emergency Brake**: Tap <kbd>Space</kbd> at any time to instantly pause the agent and hide the overlay.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **Anthropic / OpenAI / Gemini API Key** (Configured via [OpenRouter](https://openrouter.ai/) for maximum flexibility)

### Installation
1. **Clone the repo**:
   ```bash
   git clone https://github.com/robomohit1/Ai_computer.git
   cd Ai_computer
   ```
2. **Install Backend**:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
3. **Setup Environment**:
   Create a `.env` file in the root:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

### Running the Desktop App
```bash
cd desktop
npm install
npm start
```
This launches the native Electron application which automatically starts the Python backend in the background.

---

## 🏗️ Architecture

1. **The Planner**: Uses a hierarchical planning model to break high-level goals into logical sub-tasks.
2. **Safety Manager**: Every action is intercepted and analyzed. If an action is "High Risk" (like deleting system files), the agent pauses and asks for your permission via a popup in the UI.
3. **Live Screen**: In Computer Mode, the agent beams a live screenshot stream to the right-hand panel of the UI, so you can see exactly what it sees.
4. **SSE Streaming**: All thought processes, logs, and results are streamed in real-time, making the agent's logic transparent.

---

## 🛡️ Safety & Security
- **Hard-Blocked Commands**: Protects against catastrophic commands like `rm -rf /`.
- **Permission Scopes**: The agent must explicitly request `SYSTEM` permission before it can control your mouse/keyboard.
- **Isolated Browser**: Web tasks are sandboxed within a separate browser profile.

---

## 📜 License
MIT License. Created by [robomohit1](https://github.com/robomohit1).
