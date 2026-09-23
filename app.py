from flask import Flask, render_template, request
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained model
model = load_model("models/next_word_model.keras")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Convert word index to word
index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
    if index < 20000
}

SEQUENCE_LENGTH = 20


def sample_next_word(probabilities, temperature=0.8, top_k=10):

    probabilities = np.asarray(probabilities).astype("float64")

    # Get top-k word indices
    top_indices = np.argsort(probabilities)[-top_k:]

    top_probabilities = probabilities[top_indices]

    # Apply temperature
    top_probabilities = np.log(
        top_probabilities + 1e-10
    ) / temperature

    top_probabilities = np.exp(top_probabilities)

    # Normalize probabilities
    top_probabilities = (
        top_probabilities /
        np.sum(top_probabilities)
    )

    # Randomly select a word
    selected_index = np.random.choice(
        top_indices,
        p=top_probabilities
    )

    return selected_index


def generate_text(
    seed_text,
    number_of_words,
    temperature=0.8,
    top_k=10
):

    generated_text = seed_text

    for _ in range(number_of_words):

        token_list = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        token_list = token_list[-SEQUENCE_LENGTH:]

        token_list = pad_sequences(
            [token_list],
            maxlen=SEQUENCE_LENGTH,
            padding="pre"
        )

        probabilities = model.predict(
            token_list,
            verbose=0
        )[0]

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
            break

        generated_text += " " + next_word

    return generated_text


@app.route("/", methods=["GET", "POST"])
def home():

    generated_text = ""
    input_text = ""
    number_of_words = 20
    temperature = 0.8

    if request.method == "POST":

        input_text = request.form.get(
            "text",
            ""
        ).strip()

        number_of_words = int(
            request.form.get(
                "number_of_words",
                20
            )
        )

        temperature = float(
            request.form.get(
                "temperature",
                0.8
            )
        )

        if input_text:

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


if __name__ == "__main__":
    app.run(debug=True)