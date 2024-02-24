import numpy as np
import pandas as pd
from openai import OpenAI

documents = [
    "There are good people here",
    "And its nice in the right seasons",
    "Seasons are nice and good"
]

def build_client():
    api_key = "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"
    client = OpenAI(
        api_key=api_key
    )  # get API key from platform.openai.com
    return client

def text_to_vectors(text_inputs):
    client = build_client()
    MODEL = "text-embedding-3-small"
    res = client.embeddings.create(
        input=text_inputs, model=MODEL
    )
    vectors = res.data
    np_embeddings = [np.array(vector.embedding) for vector in vectors]
    return np_embeddings

text_to_vectors(documents)
