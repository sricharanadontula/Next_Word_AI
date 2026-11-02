
import os

# ============================================================
# TENSORFLOW SETTINGS
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import pickle
import tensorflow as tf
import gradio as gr

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

    temperature = max(float(temperature), 0.01)

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
        f"Starting generation: '{seed_text}' | "
        f"words={number_of_words}",
        flush=True
    )

    generated_text = seed_text

    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    print(f"Initial token list: {token_list}", flush=True)

    if not token_list:
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

        try:
            print(
                "Starting TensorFlow prediction...",
                flush=True
            )

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

        predicted_word_index = sample_next_word(
            probabilities,
            temperature=temperature,
            top_k=top_k
        )

        next_word = index_to_word.get(
            predicted_word_index,
            ""
        )

        if not next_word or next_word == "<OOV>":
            print(
                f"Skipping invalid predicted index: "
                f"{predicted_word_index}",
                flush=True
            )
            continue

        generated_text += " " + next_word

        token_list.append(predicted_word_index)

        print(f"Generated word: {next_word}", flush=True)

    print(
        f"Generation completed: {generated_text}",
        flush=True
    )

    return generated_text


# ============================================================
# GRADIO INTERFACE FUNCTION
# ============================================================

def generate_from_ui(input_text, number_of_words, temperature):

    input_text = (input_text or "").strip()

    if not input_text:
        return "Please enter some text to continue."

    number_of_words = max(
        1,
        min(int(number_of_words), 20)
    )

    temperature = max(
        0.1,
        min(float(temperature), 2.0)
    )

    try:
        generated_text = generate_text(
            input_text,
            number_of_words,
            temperature=temperature,
            top_k=10
        )

        return generated_text

    except Exception as error:

        print(f"Generation error: {error}", flush=True)

        return f"Error generating text: {error}"


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Next Word AI",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Next Word AI

        ### LSTM-Based Next Word Prediction

        Enter a sentence or phrase and let the LSTM model
        generate a continuation based on the previous words.
        """
    )

    with gr.Row():

        with gr.Column():

            input_text = gr.Textbox(
                label="Enter Your Text",
                placeholder="Example: To be or",
                lines=3
            )

            number_of_words = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Words to Generate"
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="Creativity / Temperature"
            )

            generate_button = gr.Button(
                "Generate Text",
                variant="primary"
            )

        with gr.Column():

            output_text = gr.Textbox(
                label="Generated Text",
                lines=8,
                interactive=False
            )

    gr.Examples(
        examples=[
            ["To be or"],
            ["The world is"],
            ["She was very"],
            ["Once upon a time"]
        ],
        inputs=input_text
    )

    generate_button.click(
        fn=generate_from_ui,
        inputs=[
            input_text,
            number_of_words,
            temperature
        ],
        outputs=output_text
    )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
