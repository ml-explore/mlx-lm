# Copyright © 2025 Apple Inc.

import numpy as np
from mlx_lm import load, generate
import mlx.core as mx


def retrieve(question, documents, model, tokenizer):
    # Embed the question
    question_embedding = get_embedding(question, model, tokenizer)

    # Embed all documents
    doc_embeddings = [get_embedding(doc, model, tokenizer) for doc in documents]

    # Compute cosine similarity between question and each document
    similarities = [
        cosine_similarity(question_embedding, doc_emb) for doc_emb in doc_embeddings
    ]

    # Return the document with the highest similarity score
    best_idx = int(np.argmax(similarities))
    return documents[best_idx]


def get_embedding(text, model, tokenizer):
    # Tokenize the text
    tokens = tokenizer.encode(text)
    token_array = mx.array([tokens])

    # Run a full forward pass and get the last hidden state
    # by extracting from the model's transformer layers
    hidden = model.model.embed_tokens(token_array)
    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)

    # Force computation
    mx.eval(hidden)

    # Mean pool across token dimension to get sentence embedding
    embedding = np.array(hidden[0].tolist()).mean(axis=0)
    return embedding


def cosine_similarity(a, b):
    # Compute cosine similarity between two vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # Specify the checkpoint
    checkpoint = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # Load the model and tokenizer
    model, tokenizer = load(path_or_hf_repo=checkpoint)

    # A list of documents the model does not have direct access to
    documents = [
        "MLX is an array framework for machine learning on Apple silicon, developed by Apple.",
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce energy.",
        "The Python programming language was created by Guido van Rossum and first released in 1991.",
        "Apple silicon chips use a unified memory architecture where CPU and GPU share the same memory pool.",
    ]

    # The user question
    question = "What is MLX and who made it?"

    # Retrieve the most relevant document using embedding similarity
    retrieved_doc = retrieve(question, documents, model, tokenizer)

    # Build the prompt with the retrieved context injected
    prompt = f"""Use the following context to answer the question.

Context: {retrieved_doc}

Question: {question}
Answer:"""

    # Format using the chat template
    conversation = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
    )

    # Generate the answer
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=300,
        verbose=True,
    )
