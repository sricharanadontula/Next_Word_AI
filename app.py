import os

# Force TensorFlow to use CPU on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)


# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================

print("Loading model...", flush=True)

model = load_model(
    "models/next_word_model.keras",
    compile=False
)

print("Model loaded successfully.", flush=True)


print("Loading tokenizer...", flush=True)

with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully.", flush=True)


# ============================================================
# WORD INDEX MAPPING
# ============================================================

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ============================================================
# MODEL SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20


# ============================================================
# SAMPLE NEXT WORD
# ============================================================

def sample_next_word(probabilities, temperature=0.8, top_k=10):

    probabilities = np.asarray(probabilities).astype("float64")

    # Prevent invalid values
    probabilities = np.nan_to_num(probabilities)

    # Make sure top_k is not larger than vocabulary size
    top_k = min(top_k, len(probabilities))

    # Get top-k word indices
    top_indices = np.argsort(probabilities)[-top_k:]

    top_probabilities = probabilities[top_indices]

    # Apply temperature
    top_probabilities = np.log(
        top_probabilities + 1e-10
    ) / temperature

    top_probabilities = np.exp(top_probabilities)

    # Normalize probabilities
    probability_sum = np.sum(top_probabilities)

    if probability_sum == 0:
        top_probabilities = np.ones_like(top_probabilities) / len(
            top_probabilities
        )
    else:
        top_probabilities = (
            top_probabilities / probability_sum
        )

    # Randomly select a word
    selected_index = np.random.choice(
        top_indices,
        p=top_probabilities
    )

    return int(selected_index)


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    seed_text,
    number_of_words,
    temperature=0.8,
    top_k=10
):

    print(
        f"Starting generation: '{seed_text}' | "
        f"words={number_of_words}",
        flush=True
    )

    generated_text = seed_text

    # Convert initial text to tokens only once
    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    for i in range(number_of_words):

        print(
            f"Predicting word {i + 1}/{number_of_words}",
            flush=True
        )

        # Keep only the last SEQUENCE_LENGTH tokens
        current_tokens = token_list[-SEQUENCE_LENGTH:]

        # Pad sequence
        padded_tokens = pad_sequences(
            [current_tokens],
            maxlen=SEQUENCE_LENGTH,
            padding="pre"
        )

        # Direct model inference
        # This is faster than model.predict() for a single sample
        probabilities = model(
            padded_tokens,
            training=False
        ).numpy()[0]

        # Select next word
        predicted_word_index = sample_next_word(
            probabilities,
            temperature=temperature,
            top_k=top_k
        )

        # Convert index to word
        next_word = index_to_word.get(
            predicted_word_index,
            ""
        )

        if not next_word:

            print(
                "No next word found.",
                flush=True
            )

            break

        # Add generated word
        generated_text += " " + next_word

        # Add predicted token for next iteration
        token_list.append(predicted_word_index)

        print(
            f"Generated: {next_word}",
            flush=True
        )

    print(
        "Generation completed.",
        flush=True
    )

    return generated_text


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    generated_text = ""
    input_text = ""

    # Default values
    number_of_words = 5
    temperature = 0.8

    if request.method == "POST":

        input_text = request.form.get(
            "text",
            ""
        ).strip()

        try:
            number_of_words = int(
                request.form.get(
                    "number_of_words",
                    5
                )
            )
        except (ValueError, TypeError):
            number_of_words = 5

        try:
            temperature = float(
                request.form.get(
                    "temperature",
                    0.8
                )
            )
        except (ValueError, TypeError):
            temperature = 0.8

        # Keep the request within a reasonable range
        number_of_words = max(
            1,
            min(number_of_words, 20)
        )

        if input_text:

            print(
                f"Received request: "
                f"text='{input_text}', "
                f"words={number_of_words}, "
                f"temperature={temperature}",
                flush=True
            )

            generated_text = generate_text(
                input_text,
                number_of_words,
                temperature=temperature,
                top_k=10
            )

    return render_template(
        "index.html",
        generated_text=generated_text,
        input_text=input_text,
        number_of_words=number_of_words,
        temperature=temperature
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )