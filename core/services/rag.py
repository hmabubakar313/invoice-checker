from core.services.retrieve import retrieve
from core.services.generate import generate_answer


def rag_pipeline(query):
    docs = retrieve(query)
    context = "\n".join(docs)

    answer = generate_answer(context, query)
    return answer