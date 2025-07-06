import json
import os
import re
from nltk.util import ngrams
from collections import defaultdict, Counter

class NGramPredictor:
    def __init__(self, n=2):
        self.n = n
        self.model = defaultdict(Counter)

    def preprocess(self, text):
        """🔧 Lowercase and tokenize using regex to avoid NLTK punkt errors."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)  # Simple word tokenizer
        return tokens

    def train(self, text):
        """🧠 Train the model using padded n-grams."""
        tokens = self.preprocess(text)
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        for gram in ngrams(padded, self.n):
            context, next_word = tuple(gram[:-1]), gram[-1]
            self.model[context][next_word] += 1

    def predict(self, text_input):
        """🔮 Predict next word(s) based on the last n-1 tokens."""
        tokens = self.preprocess(text_input)
        if len(tokens) < self.n - 1:
            return ["<insufficient context>"]
        context = tuple(tokens[-(self.n - 1):])
        if context in self.model:
            return [word for word, _ in self.model[context].most_common(3)]
        return ["<no suggestion>"]

    def save_model(self, filepath='custom_data.json'):
        """💾 Save the trained model to a JSON file."""
        serializable = {str(k): dict(v) for k, v in self.model.items()}
        with open(filepath, 'w') as file:
            json.dump(serializable, file)

    def load_model(self, filepath='custom_data.json'):
        """📂 Load the model from a previously saved JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                loaded = json.load(file)
                for k, v in loaded.items():
                    context = tuple(eval(k))
                    self.model[context].update(v)

def main():
    predictor = NGramPredictor(n=2)
    print("💬 Bi-gram Text Predictor - NLP Powered Console\n")

    predictor.load_model()

    while True:
        print("\n📜 Menu:")
        print("1. Train model with a new sentence")
        print("2. Predict next word")
        print("3. Add custom phrase")
        print("4. Save and exit")

        choice = input("Choose (1-4): ")

        if choice == '1':
            sentence = input("📝 Enter a training sentence: ")
            predictor.train(sentence)
            print("✅ Model trained.")

        elif choice == '2':
            text_input = input("🔍 Enter input text: ")
            suggestions = predictor.predict(text_input)
            print("👉 Predictions:", suggestions)

        elif choice == '3':
            phrase = input("➕ Enter custom phrase: ")
            predictor.train(phrase)
            print("📥 Phrase added to model.")

        elif choice == '4':
            predictor.save_model()
            print("💾 Model saved. Goodbye!")
            break

        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()