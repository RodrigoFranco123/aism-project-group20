# Multi-Agent Automated Exam & Flashcard Generator

A sequential three-agent pipeline built with LangGraph and Ollama that transforms raw study notes into validated multiple-choice questions (MCQs) and active-recall flashcards. By strictly decoupling concept extraction, question authoring, and skeptical pedagogical auditing via validated JSON handoffs, the system prevents role contamination and catches hallucinations on insufficient input material.

## Group Members
- **Rodrigo Franco**
- **Filipe Pina**
*(Intelligent Agents and Multi-Agent Systems — Group 20)*

## Installation

1. **Prerequisites:**
   - Python **3.10+**
   - [Ollama](https://ollama.com/) installed and running locally (`ollama serve`)
   - Pull the local LLM used in our experiments:
     ```bash
     ollama pull llama3.2:latest
     ```

2. **Clone and set up the Python environment:**
   ```bash
   git clone [https://github.com/](https://github.com/)RodrigoFranco123/aism-project-group20.git
   cd aism-project-group20
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt