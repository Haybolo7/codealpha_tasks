from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator, MyMemoryTranslator
import anyascii
import nltk
from langdetect import detect, DetectorFactory

# Enforce deterministic language tracking profiles
DetectorFactory.seed = 0

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

app = Flask(__name__)

# Complete ISO Language mapping ledger for beautiful detection rendering
LANGUAGE_NAMES = {
    'en': 'English', 'hi': 'Hindi (हिंदी)', 'es': 'Spanish (Español)',
    'fr': 'French (Français)', 'de': 'German (Deutsch)', 'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)', 'ja': 'Japanese (日本語)', 'ms': 'Malay',
    'ar': 'Arabic', 'ru': 'Russian', 'it': 'Italian', 'pt': 'Portuguese', 'ko': 'Korean'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_agent_task():
    data = request.get_json()
    
    text = data.get('text', '').strip()
    source_lang = data.get('source_lang', 'auto')
    target_lang = data.get('target_lang', 'en')
    service = data.get('service', 'google')
    action = data.get('action', 'translate') # Options: translate, transliterate, annotate, detect

    if not text:
        return jsonify({'error': 'Input text space cannot be empty'}), 400

    try:
        # 1. Translation System
        if action == 'translate':
            if service == 'mymemory':
                src = 'en' if source_lang == 'auto' else source_lang
                translator = MyMemoryTranslator(source=src, target=target_lang)
            else:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                
            processed_output = translator.translate(text)
            return jsonify({'result': processed_output, 'mode': 'Translation'})

        # 2. Transliteration System
        elif action == 'transliterate':
            processed_output = anyascii.anyascii(text)
            if processed_output.lower() == text.lower():
                processed_output = f"[Note: Text is already in Roman/Latin script form]\n\n{processed_output}"
            return jsonify({'result': processed_output, 'mode': 'Transliteration'})

        # 3. Morphological Structural Annotation System
        elif action == 'annotate':
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            tag_dictionary = {
                'NN': 'Noun (Singular)', 'NNS': 'Noun (Plural)', 'NNP': 'Proper Noun',
                'VB': 'Verb (Base Form)', 'VBD': 'Verb (Past Tense)', 'VBG': 'Verb (Gerund/Participle)',
                'JJ': 'Adjective', 'RB': 'Adverb', 'PRP': 'Pronoun', 'DT': 'Determiner',
                'IN': 'Preposition/Conjunction', 'CD': 'Cardinal Number'
            }
            
            report = [f"--- Linguistic Annotation Report ({len(tokens)} Tokens Identified) ---"]
            for word, tag in pos_tags:
                readable_label = tag_dictionary.get(tag, f"Structure Token ({tag})")
                report.append(f"• \"{word}\" → {readable_label}")
                
            return jsonify({'result': "\n".join(report), 'mode': 'Annotation'})

        # 4. Copied/Pasted Text Language Detection Engine
        elif action == 'detect':
            raw_code = detect(text).lower()
            readable_lang = LANGUAGE_NAMES.get(raw_code, f"Unknown/Unmapped Script ({raw_code.upper()})")
            result_string = f"--- Language Detection Report ---\n\n• Identified Language: {readable_lang}\n• Standard ISO 639-1 Code: {raw_code}"
            return jsonify({'result': result_string, 'mode': 'Language Detection'})

        else:
            return jsonify({'error': 'Invalid Agent Action Requested'}), 400

    except Exception as e:
        return jsonify({'error': f"Agent processing failure: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
