# AI Computer 🚀

An autonomous AI agent that controls your computer using natural language. 

AI Computer is a high-performance, open-source alternative to proprietary computer-use agents. It is designed to be **extensible**, **provider-agnostic**, and **production-ready**, supporting multiple LLM backends and specialized operating modes.

---

## 🔥 Key Features

### 💻 Specialized Coding Mode
- **Zero-Vision Overhead**: Optimized for software engineering. Bypasses screenshots and vision models for maximum speed and accuracy.
- **Local Toolchain**: Directly interacts with your shell, filesystem, and editors.
- **Dynamic Context**: Automatically discovers your environment (OS, user home, workspace) to ensure robust path resolution across Windows, MacOS, and Linux.

### 📡 Real-Time SSE Streaming
- **Live Dashboard**: A stunning UI that shows the agent's thought process as it happens.
- **Adaptive UI**:
  - **Thinking Indicators**: Watch the agent "think" in real-time with pulse animations.
  - **Action Cards**: Color-coded results (Success/Failure) for every tool usage.
  - **Activity Log**: High-granularity system logs streamed directly from the agent's core.

### 🛠️ Environment Discovery Tools
- **`system_info`**: Dynamic runtime discovery of OS details, home directories, and common folders (Desktop/Downloads).
- **`list_directory`**: Intelligent file exploration with emoji-labeled entries and size metadata.
- **Automatic Path Injection**: The agent "just knows" where it is. It injects local environment context directly into the planning phase.

### 🌍 Provider Agnostic
- **Multi-LLM Support**: Configurable with Anthropic (Claude), OpenAI (GPT-4), Google (Gemini), Groq, and OpenRouter.
- **Free-Tier Optimization**: Works flawlessly with free-tier models via OpenRouter (e.g., `google/gemini-2.0-flash-exp:free`).

### 🛡️ Safety & Security
- **Hard-Blocked Commands**: Protects against dangerous shell commands (`rm -f /`, `del c:\`, etc.).
- **Manual Approval Flow**: High-risk actions pause execution and wait for user confirmation via the UI.
- **Isolated Workspace**: Enforces standard operating boundaries while allowing home-directory access for specific user tasks.

### 🔍 Computer Vision (WIP)
- **Image Recognition**: Experimental `find_on_screen` tool using template matching to locate UI elements by image.
- **Optical Character Recognition (OCR)**: Integrated `pytesseract` support for extracting text from the screen to aid navigation.
- **Visual Feedback**: Real-time screenshot previews in the dashboard, allowing you to monitor the agent's work in "Computer Mode".

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js (for the frontend assets)
- Playwright (optional, for computer-vision mode)

### 2. Installation
```bash
git clone https://github.com/robomohit1/Ai_computer.git
cd Ai_computer
pip install -r requirements.txt
playwright install chromium
```

### 3. Configuration
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```
Key variables:
- `AGENT_API_KEY`: Secure your API (auto-generated if empty).
- `OPENROUTER_API_KEY`: Highly recommended for accessing a variety of models.
- `WORKSPACE_DIR`: The root for the agent's coding tasks.

### 4. Launch
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```
Open your browser to `http://localhost:8080`.

---

## 🏗️ Architecture

1. **Agent Engine**: An asynchronous event loop that processes tasks in a hierarchical plan.
2. **Planner Provider**: Converts high-level goals into sequential sub-tasks using the LLM of your choice.
3. **SSE Dispatcher**: Beams logs, screenshots, and statuses to the frontend over a persistent Server-Sent Events connection.
4. **Tool Executor**: A sandboxed execution layer for filesystem, shell, and editor interactions.

---

## 🐳 Docker Support

Deploy instantly with Docker Compose:
```bash
docker-compose up --build
```

---

## 📜 License
MIT License. Created by [robomohit1](https://github.com/robomohit1).
