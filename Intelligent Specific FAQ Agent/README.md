# 📺 Lane AI (Agentic FAQ Assistant)
It is an AI FAQ agent, powered by large language models (LLMs) that answers questions, automates tasks, and resolves customer issues. It uses natural language processing to comprehend context, reason through problems, and execute workflows without human intervention.

## ⭐ Key features
1. **Context-Driven AI Streaming** : Connects to the serverless Hugging Face Inference API via the Qwen/Qwen2.5-7B-Instruct model. The agent processes conversations by appending a highly structured background knowledge base context to ensure accurate, hallucination-free answers.
2. **Advanced Parameter Control Panel** : Includes an interactive collapsible configuration drawer allowing administrators or technical users to tweak underlying model parameters dynamically (System Prompt Context, Maximum New Tokens, Temperature, and Top-p sampling).
3. **Polished Custom User Interface** : Built using a modern dual-column canvas. It overrides standard Gradio styles with comprehensive CSS injections for dark/light mode transitions, customized buttons, specific grid scaling, container branding, and quick-click question suggestion chips.

## 🏢 Benefits for the Organization
1. *Reduced Support Burden*: Automates responses to repetitive questions, freeing up human agents for complex issues.
2. *Improved Consistency* : Enforces strict brand guidelines and compliance via a centralized system prompt.
3. *Increased User Satisfaction*: Delivers immediate, accurate, and 24/7 assistance without human intervention.

## 🚀 Tech Stack
 -> **Frontend/Interface**: Gradio 6.x (gr.Blocks, gr.Group, gr.Timer, gr.Chatbot)
 
 -> **Backend Logic**: Python 3.13, datetime
 
 -> **LLM & Hosting**: Hugging Face Hub Client API (Qwen2.5-7B-Instruct), Hugging Face Spaces (CPU-basic tier)
 
 -> **Styling**: Custom CSS and Vanilla JavaScript (for dark/light mode client transitions)
 
