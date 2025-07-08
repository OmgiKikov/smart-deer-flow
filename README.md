<div align="center">

# 🦌 SmartDeerFlow

**AI-Powered Deep Research Framework with Multi-Agent Collaboration**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hqzhon/smart-deer-flow?style=social)](https://github.com/hqzhon/smart-deer-flow/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hqzhon/smart-deer-flow?style=social)](https://github.com/hqzhon/smart-deer-flow/network/members)
[![GitHub issues](https://img.shields.io/github/issues/hqzhon/smart-deer-flow)](https://github.com/hqzhon/smart-deer-flow/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/hqzhon/smart-deer-flow)](https://github.com/hqzhon/smart-deer-flow/commits/main)

[English](./README.md) | [简体中文](./README_zh.md)

</div>

## 🚀 Overview

**SmartDeerFlow** is a community-driven AI research framework that combines **Large Language Models**, **Multi-Agent Systems**, and **Advanced Tools** for automated research, content generation, and data analysis.

**Key Highlights:**
- 🤖 **Multi-Agent Collaboration** - Intelligent task distribution and coordination
- ⚡ **Performance Optimized** - Advanced parallel processing and caching
- 🔍 **Multi-Source Search** - Tavily, Brave, DuckDuckGo, ArXiv integration
- 📊 **Rich Output Formats** - Reports, Podcasts, Presentations
- 🌐 **Web & Console UI** - Flexible interaction modes
- 🔗 **Extensible Architecture** - MCP and RAG integrations

> Forked from [DeerFlow](https://github.com/bytedance/deer-flow) with enhanced features and community-driven improvements.



## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🌟 Features](#-features)
- [⚡ Performance](#-performance)
- [🏗️ Architecture](#-architecture)
- [📚 Examples](#-examples)
- [🐳 Docker](#-docker)
- [🛠️ Development](#-development)
- [❓ FAQ](#-faq)
- [📜 License](#-license)

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** and **Node.js 22+**
- **Recommended:** [`uv`](https://docs.astral.sh/uv/) for Python, [`pnpm`](https://pnpm.io/) for Node.js

### Installation

```bash
# 1. Clone and setup
git clone https://github.com/hqzhon/smart-deer-flow.git
cd smart-deer-flow
uv sync

# 2. Configure API keys
cp .env.example .env          # Add your API keys (Tavily, Brave, etc.)
cp conf.yaml.example conf.yaml  # Configure LLM settings

# 3. Optional: Install additional tools
brew install marp-cli         # For PPT generation
cd web && pnpm install        # For Web UI
```

### Usage

```bash
# Console Mode (Quick Start)
uv run main.py "What is quantum computing?"

# Interactive Mode
uv run main.py --interactive

# Web UI Mode
./bootstrap.sh -d  # macOS/Linux
# Visit http://localhost:3000
```

> 📖 **Configuration:** See [Configuration Guide](docs/configuration_guide.md) for detailed setup instructions.

## Supported Search Engines

SmartDeerFlow supports multiple search engines that can be configured in your `.env` file using the `SEARCH_API` variable:

- **Tavily** (default): A specialized search API for AI applications

  - Requires `TAVILY_API_KEY` in your `.env` file
  - Sign up at: https://app.tavily.com/home

- **DuckDuckGo**: Privacy-focused search engine

  - No API key required

- **Brave Search**: Privacy-focused search engine with advanced features

  - Requires `BRAVE_SEARCH_API_KEY` in your `.env` file
  - Sign up at: https://brave.com/search/api/

- **Arxiv**: Scientific paper search for academic research
  - No API key required
  - Specialized for scientific and academic papers

To configure your preferred search engine, set the `SEARCH_API` variable in your `.env` file:

```bash
# Choose one: tavily, duckduckgo, brave_search, arxiv
SEARCH_API=tavily
```

## ⚡ Performance

### Optimization Levels
| Level | Workers | Features |
|-------|---------|----------|
| **Basic** | 4 | Parallel processing |
| **Standard** | 8 | Rate limiting, caching |
| **Advanced** | 12 | Intelligent scheduling |
| **Maximum** | 16 | ML-driven optimization |

### Quick Setup
```bash
# Enable performance mode
cp .env.performance.example .env.performance
export DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION=true
python scripts/start_server.py --performance-mode performance
```

📖 **Detailed Guides:** [Performance](./README_PERFORMANCE.md) | [Parallel Optimization](./README_PARALLEL_OPTIMIZATION.md)

## 🌟 Features

### 🤖 AI & LLM Integration
- **Multi-Model Support** - OpenAI, Anthropic, Qwen, and more via [LiteLLM](https://docs.litellm.ai/docs/providers)
- **Smart Agent Coordination** - Dynamic task distribution and collaboration
- **Context-Aware Processing** - Intelligent content understanding and generation

### 🔍 Research & Data Collection
- **Multi-Source Search** - Tavily, Brave, DuckDuckGo, ArXiv integration
- **Web Crawling** - Advanced content extraction with Jina
- **RAG Integration** - Private knowledge base support via [RAGFlow](https://github.com/infiniflow/ragflow)
- **MCP Extensions** - Expandable tool ecosystem

### ⚡ Performance & Optimization
- **Parallel Processing** - 4-16 worker configurations with intelligent scheduling
- **Hierarchical Caching** - Multi-level cache system (L1/L2/L3)
- **Adaptive Rate Limiting** - Dynamic request management
- **Smart Error Recovery** - Automatic retry with exponential backoff

### 📊 Content Generation
- **Research Reports** - Comprehensive analysis and documentation
- **Podcast Scripts** - AI-powered audio content generation
- **Presentations** - Automated PowerPoint creation
- **Interactive Editing** - Notion-style block editing with AI assistance

### 🤝 Human Collaboration
- **Human-in-the-Loop** - Interactive plan review and modification
- **Real-time Feedback** - Natural language plan editing
- **Consensus Systems** - Multi-agent decision making
- **Role-based Access** - Dynamic permission management

## Architecture

DeerFlow implements a modular multi-agent system architecture designed for automated research and code analysis. The system is built on LangGraph, enabling a flexible state-based workflow where components communicate through a well-defined message passing system.

![Architecture Diagram](./assets/architecture.png)

> See it live at [deerflow.tech](https://deerflow.tech/#multi-agent-architecture)

The system employs a streamlined workflow with the following components:

1. **Coordinator**: The entry point that manages the workflow lifecycle

   - Initiates the research process based on user input
   - Delegates tasks to the planner when appropriate
   - Acts as the primary interface between the user and the system

2. **Planner**: Strategic component for task decomposition and planning

   - Analyzes research objectives and creates structured execution plans
   - Determines if enough context is available or if more research is needed
   - Manages the research flow and decides when to generate the final report

3. **Research Team**: A collection of specialized agents that execute the plan:

   - **Researcher**: Conducts web searches and information gathering using tools like web search engines, crawling and even MCP services.
   - **Coder**: Handles code analysis, execution, and technical tasks using Python REPL tool.
     Each agent has access to specific tools optimized for their role and operates within the LangGraph framework

4. **Reporter**: Final stage processor for research outputs
   - Aggregates findings from the research team
   - Processes and structures the collected information
   - Generates comprehensive research reports

## Text-to-Speech Integration

DeerFlow now includes a Text-to-Speech (TTS) feature that allows you to convert research reports to speech. This feature uses the volcengine TTS API to generate high-quality audio from text. Features like speed, volume, and pitch are also customizable.

### Using the TTS API

You can access the TTS functionality through the `/api/tts` endpoint:

```bash
# Example API call using curl
curl --location 'http://localhost:8000/api/tts' \
--header 'Content-Type: application/json' \
--data '{
    "text": "This is a test of the text-to-speech functionality.",
    "speed_ratio": 1.0,
    "volume_ratio": 1.0,
    "pitch_ratio": 1.0
}' \
--output speech.mp3
```

## 🛠️ Development

### Quick Commands
```bash
# Testing
pytest tests/ --cov=deer_flow

# Code Quality
ruff format . && ruff check .

# Debug with LangGraph Studio
# Install: https://github.com/langchain-ai/langgraph-studio
```

### Debugging with LangGraph Studio

DeerFlow uses LangGraph for its workflow architecture. You can use LangGraph Studio to debug and visualize the workflow in real-time.

#### Running LangGraph Studio Locally

DeerFlow includes a `langgraph.json` configuration file that defines the graph structure and dependencies for the LangGraph Studio. This file points to the workflow graphs defined in the project and automatically loads environment variables from the `.env` file.

##### Mac

```bash
# Install uv package manager if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
```

##### Windows / Linux

```bash
# Install dependencies
pip install -e .
pip install -U "langgraph-cli[inmem]"

# Start the LangGraph server
langgraph dev
```

After starting the LangGraph server, you'll see several URLs in the terminal:

- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API Docs: http://127.0.0.1:2024/docs

Open the Studio UI link in your browser to access the debugging interface.

#### Using LangGraph Studio

In the Studio UI, you can:

1. Visualize the workflow graph and see how components connect
2. Trace execution in real-time to see how data flows through the system
3. Inspect the state at each step of the workflow
4. Debug issues by examining inputs and outputs of each component
5. Provide feedback during the planning phase to refine research plans

When you submit a research topic in the Studio UI, you'll be able to see the entire workflow execution, including:

- The planning phase where the research plan is created
- The feedback loop where you can modify the plan
- The research and writing phases for each section
- The final report generation

### Enabling LangSmith Tracing

DeerFlow supports LangSmith tracing to help you debug and monitor your workflows. To enable LangSmith tracing:

1. Make sure your `.env` file has the following configurations (see `.env.example`):

   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   LANGSMITH_API_KEY="xxx"
   LANGSMITH_PROJECT="xxx"
   ```

2. Start tracing and visualize the graph locally with LangSmith by running:
   ```bash
   langgraph dev
   ```

This will enable trace visualization in LangGraph Studio and send your traces to LangSmith for monitoring and analysis.

## 🐳 Docker

```bash
# Quick start with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d
```

**Access:** http://localhost:3000

**Includes:** Backend API + Frontend UI + Data persistence

## 📚 Examples

### Research Reports
```bash
# Generate comprehensive research report
uv run main.py "AI impact on healthcare"

# Custom planning parameters
uv run main.py --max_plan_iterations 3 "Quantum computing impact"
```

### Interactive Mode
```bash
# Interactive session with built-in questions
uv run main.py --interactive

# Basic interactive prompt
uv run main.py
```

### Sample Reports
- [OpenAI Sora Analysis](examples/openai_sora_report.md)
- [Agent to Agent Protocol](examples/what_is_agent_to_agent_protocol.md)
- [Bitcoin Price Analysis](examples/bitcoin_price_fluctuation.md)
- [AI in Healthcare](examples/AI_adoption_in_healthcare.md)
- [Quantum Cryptography](examples/Quantum_Computing_Impact_on_Cryptography.md)

## 🔧 Command Line Options

```bash
# Basic options
uv run main.py "Your research question"
uv run main.py --interactive
uv run main.py --enable-human-in-loop "Your question"

# Performance tuning
uv run main.py --performance-mode advanced --workers 12
uv run main.py --enable-caching --max_plan_iterations 5

# Output formats
uv run main.py --output-format report "Your question"

# View all options
uv run main.py --help
```

### Interactive Mode

The application supports an interactive mode with built-in questions in both English and Chinese:

1. Launch the interactive mode:
   ```bash
   uv run main.py --interactive
   ```

2. Select your preferred language (English or 中文)
3. Choose from a list of built-in questions or ask your own question
4. The system will process your question and generate a comprehensive research report

### Human in the Loop

SmartDeerFlow includes a human in the loop mechanism that allows you to review, edit, and approve research plans before they are executed:

1. **Plan Review**: When enabled, the system presents the generated research plan for your review
2. **Providing Feedback**: Accept with `[ACCEPTED]` or edit with `[EDIT PLAN] Your feedback`
3. **API Integration**: Use the `feedback` parameter in API calls to provide plan modifications

## ❓ FAQ

**Q: How do I configure API keys?**  
A: Copy `.env.example` to `.env` and add your keys. See [Configuration Guide](docs/configuration_guide.md).

**Q: Can I use local models?**  
A: Yes, supports Ollama and other local providers via `.env` configuration.

**Q: How to enable performance optimization?**  
A: Set `DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION=true` and use `--performance-mode advanced`.

**Q: What search engines are supported?**  
A: Tavily, Brave, DuckDuckGo, ArXiv, and more via MCP integrations.

**Q: How to contribute?**  
A: Fork → Make changes → Submit PR. Check contribution guidelines.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

**Built with:** [LangChain](https://langchain.com/) • [LangGraph](https://langchain-ai.github.io/langgraph/) • [FastAPI](https://fastapi.tiangolo.com/)  
**Forked from:** [DeerFlow](https://github.com/bytedance/deer-flow) by ByteDance  
**Thanks to:** All contributors and the open-source community
