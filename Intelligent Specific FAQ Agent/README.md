# 📺 Lane AI (Agentic FAQ Assistant)
It is an AI FAQ agent, powered by large language models (LLMs) that answers questions, automates tasks, and resolves customer issues. It uses natural language processing to comprehend context, reason through problems, and execute workflows without human intervention.

## ⭐ Key features
1. **Context-Driven AI Streaming** : Connects to the serverless Hugging Face Inference API via the Qwen/Qwen2.5-7B-Instruct model. The agent processes conversations by appending a highly structured background knowledge base context to ensure accurate, hallucination-free answers.
2. **Advanced Parameter Control Panel** : Includes an interactive collapsible configuration drawer allowing administrators or technical users to tweak underlying model parameters dynamically (System Prompt Context, Maximum New Tokens, Temperature, and Top-p sampling).
3. **NLP Preprocessing**: Incoming user text is automatically forced lowercase, stripped of noise using regular expressions, and split into clean grammatical components via NLTK word tokenization.
4. **Similarity Matching**: The application creates a TF-IDF (Term Frequency-Inverse Document Frequency) Vector Matrix out of all stored questions. When you send a message, it computes the Cosine Similarity between your tokenized query vector and the knowledge base matrix to instantly extract the closest matching FAQ pair mathematically.

## 🏢 Benefits for the Organization
1. *Reduced Support Burden*: Automates responses to repetitive questions, freeing up human agents for complex issues.
2. *Improved Consistency* : Enforces strict brand guidelines and compliance via a centralized system prompt.
3. *Increased User Satisfaction*: Delivers immediate, accurate, and 24/7 assistance without human intervention.

## 🚀 Tech Stack
The project was made utilising the below given technologies. The project can be accessed using the given URL: https://huggingface.co/spaces/VectorV17/Intelli_Agent 

 -> **Frontend/Interface**: Gradio 6.x (gr.Blocks, gr.Group, gr.Timer, gr.Chatbot)
 
 -> **Backend Logic**: Python 3.13, datetime 
 
 -> **Packages & Hosting**: nltk library, scikit-learn, Hugging Face Spaces (CPU-basic tier)
 
 -> **Styling**: Custom CSS and Vanilla JavaScript (for dark/light mode client transitions)
 
