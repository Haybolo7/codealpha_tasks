import gradio as gr
from huggingface_hub import InferenceClient
from datetime import datetime
import re

# Import NLP and Math Matching frameworks
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK tokenization assets safely on basic CPU bootup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass  # Fallback to standard regex splitting if environment is strictly offline

# Initialize the Hugging Face Serverless client
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

# 1. Collect FAQs related to topics/products into a structured corpus
FAQ_DATA = [
    {
        "q": "What are your shipping times and policies?",
        "a": "Shipping times vary by location. Typically, local retail deliveries take 2-3 business days, while international dropshipping options take 7-14 business days. All tracking codes are sent via email."
    },
    {
        "q": "What is the return policy for products?",
        "a": "We offer a flexible 30-day return policy. Items must be unworn, in their original packaging layout, and accompanied by the checkout receipt."
    },
    {
        "q": "What are your service project timelines and quotes?",
        "a": "Standard design and consulting project timelines range between 2 to 6 weeks. You can request a personalized pricing quote directly via our booking panel."
    },
    {
        "q": "What are the differences between your software subscription tiers?",
        "a": "We offer Free, Pro, and Enterprise tiers. Pro unlocks comprehensive API documentation access, and Enterprise unlocks custom third-party platform integrations."
    },
    {
        "q": "How do I unlock my course completion certificate?",
        "a": "Completion certificates unlock automatically on your student profile dashboard as soon as all video modules and the final evaluation quiz are marked 100% complete."
    },
    {
        "q": "What insurance providers do you accept at the clinic?",
        "a": "We accept major national medical insurance providers. Please contact our front office coordinator to verify your specific policy layout before scheduling appointments."
    },
    {
        "q": "What are your standard check-in and check-out times?",
        "a": "Standard hospitality check-in opens at 3:00 PM, and checkout must be finalized by 11:00 AM. Early check-in requests can be accommodated based on live room availability."
    }
]

def perform_nlp_cosine_matching(user_query):
    """
    Preprocesses text using tokenization and applies Cosine Similarity 
    matching over TF-IDF vectors to find the best matching FAQ.
    """
    try:
        tokens = nltk.word_tokenize(user_query.lower())
        cleaned_query = " ".join([w for w in tokens if re.match(r'^\w+$', w)])
    except Exception:
        cleaned_query = " ".join(re.findall(r'\w+', user_query.lower()))
        
    if not cleaned_query.strip():
        return None, 0.0

    questions_corpus = [faq["q"] for faq in FAQ_DATA]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions_corpus)
    
    query_vector = vectorizer.transform([cleaned_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]
    
    return FAQ_DATA[best_match_idx], float(highest_score)

def get_thematic_datetime():
    """Generates precise, real-time updated date and running seconds strings."""
    now = datetime.now()
    day_int = now.day
    if 11 <= day_int <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_int % 10, "th")
        
    formatted_date = f"{now.strftime('%d')}{suffix} {now.strftime('%B %Y')} | Time: {now.strftime('%I:%M:%S %p').lower()}"
    return f"<div style='text-align: right; font-weight: 600; font-size: 14px; font-family: monospace; letter-spacing: 0.5px;'>{formatted_date}</div>"

def extract_text(content):
    """Safely extracts plain string text from any Gradio 6 message structure."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        extracted_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                extracted_parts.append(block["text"])
            elif isinstance(block, str):
                extracted_parts.append(block)
        return "".join(extracted_parts)
    if isinstance(content, dict):
        if "text" in content: return content["text"]
        if "content" in content: return content["content"]
    return str(content)

def user_side(user_message, history):
    """Handles incoming user messages and sanitizes state history format."""
    if not user_message.strip():
        return "", history
    
    clean_history = []
    for msg in history:
        if isinstance(msg, dict):
            clean_history.append({"role": msg["role"], "content": extract_text(msg["content"])})
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            clean_history.append({"role": msg.role, "content": extract_text(msg.content)})
            
    clean_history.append({"role": "user", "content": user_message})
    return "", clean_history

def bot_side(history, system_message, max_tokens, temperature, top_p):
    """Streams responses out, embedding the NLP similarity match directly into context."""
    if not history:
        yield history
        return

    user_query = extract_text(history[-1]["content"])
    matched_faq, match_score = perform_nlp_cosine_matching(user_query)
    
    faq_context_injection = ""
    
    # RECTIFIED LOGIC: The metadata line only outlines telemetry metrics. It doesn't dump the duplicate answer text block.
    if matched_faq and match_score > 0.20:
        nlp_metadata_prefix = f"🔍 *NLP Intent Match: \"{matched_faq['q']}\" (Confidence: {match_score*100:.1f}%)*\n\n"
        faq_context_injection = (
            f"The user's query closely matches this curated FAQ entry:\n"
            f"Question: {matched_faq['q']}\n"
            f"Answer: {matched_faq['a']}\n"
            f"Please output this answer clearly and conversationally. Do not wrap it in quotes."
        )
    else:
        nlp_metadata_prefix = "🔍 *NLP Search Notice: No close database FAQ match found. Processing via general knowledge base...*\n\n"
        faq_context_injection = "No matching database entry found. Answer the question directly using standard general system knowledge."

    full_system_prompt = (
        f"{system_message}\n\n"
        f"{faq_context_injection}"
    )
    
    messages = [{"role": "system", "content": full_system_prompt}]
    clean_history = []
    
    for msg in history[:-1]:
        if isinstance(msg, dict):
            role, content = msg.get("role"), extract_text(msg.get("content"))
        else:
            role, content = getattr(msg, "role", None), extract_text(getattr(msg, "content", ""))
        if role and content:
            messages.append({"role": role, "content": content})
            clean_history.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})
    clean_history.append({"role": "user", "content": user_query})
    
    # Pre-populate the visualization header block securely
    clean_history.append({"role": "assistant", "content": nlp_metadata_prefix})

    try:
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=int(max_tokens),
            stream=True,
            temperature=float(temperature),
            top_p=float(top_p),
        ):
            token = message_chunk.choices[0].delta.content
            if token is not None:
                clean_history[-1]["content"] += token
                yield clean_history
    except Exception as e:
        clean_history[-1]["content"] += f"\nAn error occurred while connecting to the model API: {str(e)}"
        yield clean_history

# Custom UI Theme Stylesheet
custom_css = """
#header-container {
    background-color: #eef2f5 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin-bottom: 20px !important;
    border: 1px solid #dcdfe6 !important;
}
body.dark #header-container {
    background-color: #1f2937 !important;
    border-color: #374151 !important;
}
#header-container * { color: #1a1a1a !important; }
body.dark #header-container * { color: #f9fafb !important; }
#right-meta-card {
    background-color: #005f53 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    color: white !important;
    text-align: center;
}
#right-meta-card * { color: white !important; }
#send-btn {
    background-color: #ff6600 !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
}
#toggle-mode-btn {
    background-color: #4b5563 !important;
    color: white !important;
    border-radius: 20px !important;
    cursor: pointer;
    font-size: 13px !important;
}
.meta-status-text {
    text-align: left; 
    font-size: 13px; 
    line-height: 1.8; 
    margin-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.2);
    padding-top: 15px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    
    # 1. Header Section
    with gr.Row(elem_id="header-container"):
        with gr.Column(scale=3):
            gr.Markdown(
                """
                ### About Us
                At FAQ Agentic Hub, we are passionate about the power of structured, NLP-driven automation and its ability to transform customer workflows. This engine utilizes real-time Tokenization and Cosine Similarity computations.
                """
            )
        with gr.Column(scale=1):
            time_display = gr.HTML(value=get_thematic_datetime())
            toggle_btn = gr.Button("☀️/🌙 ➔ Change Mode (Day / Night)", elem_id="toggle-mode-btn")
            toggle_btn.click(None, None, None, js="() => document.body.classList.toggle('dark')")

    # 2. Workspace Layout
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Lane NLP AI Support Stream", 
                value=[{"role": "assistant", "content": "Hello! I'm Lane, your upgraded NLP FAQ Support Agent. Ask me questions about shipping, returns, or certificates, and watch the cosine-similarity system extract matches in real-time."}]
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask Lane anything regarding automated FAQ policies...", 
                    show_label=False, 
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")
            
            with gr.Row():
                suggest_btn1 = gr.Button("What are your shipping times and policies?")
                suggest_btn2 = gr.Button("How do I unlock my course completion certificate?")

        with gr.Column(scale=1):
            with gr.Group(elem_id="right-meta-card"):
                gr.HTML(
                    """
                    <div style='display: flex; justify-content: center; margin-bottom: 12px;'>
                        <div style='background: white; border-radius: 50%; padding: 12px; width: 110px; height: 110px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <img src='https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/robot/default/48px.svg' style='width: 70px; filter: invert(27%) sepia(91%) saturate(2853%) hue-rotate(152deg) brightness(93%) contrast(101%);'>
                        </div>
                    </div>
                    <h2 style='margin: 5px 0 2px 0; font-size: 24px;'>Lane <span style='font-size: 11px; background: rgba(255,255,255,0.25); padding: 2px 7px; border-radius: 4px; vertical-align: middle;'>NLP AI</span></h2>
                    <p style='margin: 0; opacity: 0.85; font-size: 14px;'>Similarity-Augmented Assistant</p>
                    
                    <div class='meta-status-text'>
                        <p>🟢 NLP Match Engine Online</p>
                        <p><strong>Vectorization Model:</strong> Text TF-IDF Vectorizer</p>
                        <p><strong>Processing Hardware:</strong> Base CPU Node</p>
                    </div>
                    """
                )
                
            with gr.Accordion("Advanced Agent Parameter Configurations", open=False):
                system_msg = gr.Textbox(
                    value="You are an expert AI FAQ Agent assistant. Answer user queries concisely, expanding intelligently upon matched context summaries when available.", 
                    label="System Prompt Context Injection"
                )
                max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
                temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.4, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (Nucleus Sampling)")

    # 3. Components Event Wiring
    send_btn.click(
        user_side, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False
    ).then(
        bot_side, inputs=[chatbot, system_msg, max_tokens, temperature, top_p], outputs=[chatbot]
    )
    
    msg_input.submit(
        user_side, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False
    ).then(
        bot_side, inputs=[chatbot, system_msg, max_tokens, temperature, top_p], outputs=[chatbot]
    )

    suggest_btn1.click(
        user_side, inputs=[suggest_btn1, chatbot], outputs=[msg_input, chatbot], queue=False
    ).then(
        bot_side, inputs=[chatbot, system_msg, max_tokens, temperature, top_p], outputs=[chatbot]
    )
    
    suggest_btn2.click(
        user_side, inputs=[suggest_btn2, chatbot], outputs=[msg_input, chatbot], queue=False
    ).then(
        bot_side, inputs=[chatbot, system_msg, max_tokens, temperature, top_p], outputs=[chatbot]
    )

    # Running digital time clock update loop
    clock_timer = gr.Timer(1.0)
    clock_timer.tick(fn=get_thematic_datetime, outputs=time_display, queue=False)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default())
