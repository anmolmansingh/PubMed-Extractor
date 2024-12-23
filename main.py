from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from utils import encode_pdf
from helper_functions import *

# from evaluation.evalute_rag import *

app = FastAPI()

try:
    if load_dotenv(".env") is False:
        raise TypeError
except TypeError:
    print("Unable to load .env file.")
    quit()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL_NAME = "gpt-4o"


class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask_llm(query: Query):
    chunks_vector_store = encode_pdf("papers/3.pdf", chunk_size=1000, chunk_overlap=200)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    context = retrieve_context_per_question(query, chunks_query_retriever)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        # temperature=TEMPERATURE,
        max_tokens=200,
        messages=[
            #   {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.question}
        ],
    )
    # response = openai.Completion.create(
    #     engine="gpt-4o",
    #     prompt=query.question,
    #     max_tokens=200
    # )
    # show_context(context)
    return {"answer": context}
    # return {"answer": completion.choices[0].message.content}


@app.post("/rag")
async def rag_pipeline(query: Query):

    context = retrieve_relevant_cuntext(query.question, index, sentences)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        # temperature=TEMPERATURE,
        max_tokens=200,
        messages=[
            #   {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context: {context}\nAnswer the question: {query.question}",
            }
        ],
    )
    # response = openai.Completion.create(
    #     engine="gpt-4o",
    #     prompt=query.question,
    #     max_tokens=200
    # )
    return {"answer": completion.choices[0].message.content}
