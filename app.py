import os

# ============================================================
# RENDER / TENSORFLOW SETTINGS
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

SEQUENCE_LENGTH = 20


# ============================================================
# LOAD SMALL MODEL
# ============================================================

print("========================================", flush=True)
print("Loading small LSTM model...", flush=True)
print("========================================", flush=True)

model = load_model(
    "models/next_word_model_small.keras",
    compile=False
)

print("Small model loaded successfully.", flush=True)
print(f"Model input shape: {model.input_shape}", flush=True)
print(f"Model output shape: {model.output_shape}", flush=True)


# ============================================================
# LOAD SMALL TOKENIZER
# ============================================================

print("Loading small tokenizer...", flush=True)

with open("models/tokenizer_small.pkl", "rb") as file:
    tokenizer = pickle.load(file)

print("Small tokenizer loaded successfully.", flush=True)
print(f"Vocabulary size: {len(tokenizer.word_index)}", flush=True)


# ============================================================
# WORD INDEX MAPPING
# ============================================================

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ============================================================
# SAMPLE NEXT WORD
# ============================================================

def sample_next_word(probabilities, temperature=0.8, top_k=10):

    probabilities = np.asarray(
        probabilities,
        dtype="float64"
    )

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Never select <OOV>
    if len(probabilities) > 1:
        probabilities[1] = 0.0

    top_k = max(
        1,
        min(int(top_k), len(probabilities))
    )

    top_indices = np.argsort(probabilities)[-top_k:]

    top_probabilities = probabilities[top_indices]

    temperature = max(
        float(temperature),
        0.01
    )

    top_probabilities = np.log(
        top_probabilities + 1e-10
    ) / temperature

    top_probabilities = np.exp(top_probabilities)

    probability_sum = np.sum(top_probabilities)

    if (
        probability_sum <= 0
        or not np.isfinite(probability_sum)
    ):
        top_probabilities = (
            np.ones_like(top_probabilities)
            / len(top_probabilities)
        )
    else:
        top_probabilities = (
            top_probabilities / probability_sum
        )

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
        f"'{seed_text}' | words={number_of_words}",
        flush=True
    )

    generated_text = seed_text

    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    print(
        f"Initial token list: {token_list}",
        flush=True
    )

    if not token_list:
        print(
            "No known words found in tokenizer.",
            flush=True
        )
        return seed_text

    for i in range(number_of_words):

        print(
            f"Predicting word {i + 1}/{number_of_words}",
            flush=True
        )

        current_tokens = token_list[-SEQUENCE_LENGTH:]

        single_input = pad_sequences(
            [current_tokens],
            maxlen=SEQUENCE_LENGTH,
            padding="pre"
        )

        print(
            f"Model input shape: {single_input.shape}",
            flush=True
        )

        try:

            print(
                "Starting TensorFlow prediction...",
                flush=True
            )

            # New small model accepts (1, 20) directly.
            predictions = model.predict(
                single_input,
                verbose=0
            )

            print(
                "TensorFlow prediction completed.",
                flush=True
            )

        except Exception as error:

            print(
                f"MODEL PREDICTION ERROR: "
                f"{type(error).__name__}: {error}",
                flush=True
            )

            raise

        probabilities = predictions[0]

        print(
            f"Prediction output shape: "
            f"{probabilities.shape}",
            flush=True
        )

        predicted_word_index = sample_next_word(
            probabilities,
            temperature=temperature,
            top_k=top_k
        )

        next_word = index_to_word.get(
            predicted_word_index,
            ""
        )

        if (
            not next_word
            or next_word == "<OOV>"
        ):
            print(
                f"Skipping invalid predicted index: "
                f"{predicted_word_index}",
                flush=True
            )
            continue

        generated_text += " " + next_word

        token_list.append(
            predicted_word_index
        )

        print(
            f"Generated word: {next_word}",
            flush=True
        )

    print(
        f"Generation completed: {generated_text}",
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

        number_of_words = max(
            1,
            min(number_of_words, 20)
        )

        temperature = max(
            0.1,
            min(temperature, 2.0)
        )

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