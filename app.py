import os

# ============================================================
# RENDER / TENSORFLOW SETTINGS
# ============================================================

# Force TensorFlow to use CPU on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Reduce unnecessary TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from flask import Flask, render_template, request
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)


# ============================================================
# MODEL SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...", flush=True)

model = load_model(
    "models/next_word_model.keras",
    compile=False
)

print("Model loaded successfully.", flush=True)

print(
    f"Model input shape: {model.input_shape}",
    flush=True
)

print(
    f"Model output shape: {model.output_shape}",
    flush=True
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

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


print(
    f"Tokenizer vocabulary size: {len(tokenizer.word_index)}",
    flush=True
)


# ============================================================
# SAMPLE NEXT WORD
# ============================================================

def sample_next_word(
    probabilities,
    temperature=0.8,
    top_k=10
):

    probabilities = np.asarray(
        probabilities
    ).astype("float64")

    # Remove invalid values
    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Make sure top_k is valid
    top_k = min(
        top_k,
        len(probabilities)
    )

    # Get top-k word indices
    top_indices = np.argsort(
        probabilities
    )[-top_k:]

    top_probabilities = probabilities[
        top_indices
    ]

    # Apply temperature
    top_probabilities = np.log(
        top_probabilities + 1e-10
    ) / temperature

    top_probabilities = np.exp(
        top_probabilities
    )

    # Normalize probabilities
    probability_sum = np.sum(
        top_probabilities
    )

    if probability_sum <= 0:

        top_probabilities = (
            np.ones_like(top_probabilities)
            / len(top_probabilities)
        )

    else:

        top_probabilities = (
            top_probabilities
            / probability_sum
        )

    # Select a word
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
        f"Starting generation: "
        f"'{seed_text}' | "
        f"words={number_of_words}",
        flush=True
    )

    generated_text = seed_text

    # Convert seed text into tokens
    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    print(
        f"Initial token list: {token_list}",
        flush=True
    )

    # If no known words were found
    if not token_list:

        print(
            "No known words found in tokenizer.",
            flush=True
        )

        return seed_text

    for i in range(number_of_words):

        print(
            f"Predicting word "
            f"{i + 1}/{number_of_words}",
            flush=True
        )

        # Keep only the last 20 tokens
        current_tokens = token_list[
            -SEQUENCE_LENGTH:
        ]

        # Convert to shape (1, 20)
        padded_tokens = pad_sequences(
            [current_tokens],
            maxlen=SEQUENCE_LENGTH,
            padding="pre"
        )

        print(
            f"Input shape: "
            f"{padded_tokens.shape}",
            flush=True
        )

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            probabilities = model.predict(
                padded_tokens,
                verbose=0
            )[0]

        except Exception as error:

            print(
                f"Prediction error: {error}",
                flush=True
            )

            raise error

        print(
            "Prediction completed.",
            flush=True
        )

        # ====================================================
        # SELECT NEXT WORD
        # ====================================================

        predicted_word_index = sample_next_word(
            probabilities,
            temperature=temperature,
            top_k=top_k
        )

        next_word = index_to_word.get(
            predicted_word_index,
            ""
        )

        if not next_word:

            print(
                f"No word found for index "
                f"{predicted_word_index}",
                flush=True
            )

            break

        # Add generated word
        generated_text += " " + next_word

        # Add token for next prediction
        token_list.append(
            predicted_word_index
        )

        print(
            f"Generated word: {next_word}",
            flush=True
        )

    print(
        f"Generation completed: "
        f"{generated_text}",
        flush=True
    )

    return generated_text


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
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

        # ----------------------------------------------------
        # Number of words
        # ----------------------------------------------------

        try:

            number_of_words = int(
                request.form.get(
                    "number_of_words",
                    5
                )
            )

        except (
            ValueError,
            TypeError
        ):

            number_of_words = 5

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        try:

            temperature = float(
                request.form.get(
                    "temperature",
                    0.8
                )
            )

        except (
            ValueError,
            TypeError
        ):

            temperature = 0.8

        # ----------------------------------------------------
        # Safety limits
        # ----------------------------------------------------

        number_of_words = max(
            1,
            min(
                number_of_words,
                20
            )
        )

        temperature = max(
            0.1,
            min(
                temperature,
                2.0
            )
        )

        # ----------------------------------------------------
        # Generate text
        # ----------------------------------------------------

        if input_text:

            print(
                "========================================",
                flush=True
            )

            print(
                f"Received request: "
                f"text='{input_text}', "
                f"words={number_of_words}, "
                f"temperature={temperature}",
                flush=True
            )

            print(
                "========================================",
                flush=True
            )

            generated_text = generate_text(
                input_text,
                number_of_words,
                temperature=temperature,
                top_k=10
            )

    # --------------------------------------------------------
    # Render page
    # --------------------------------------------------------

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

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )