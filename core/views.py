from rest_framework.decorators import api_view
from rest_framework.response import Response
from core.services.rag import rag_pipeline
from core.services.ingest import ingest_document
from core.services.orchestrator import decide_and_act
from core.serializers import DocumentSerializer


@api_view(['POST'])
def ask_question(request):
    query = request.data.get("query")
    answer = rag_pipeline(query)
    return Response({"answer": answer})


@api_view(['POST'])
def upload_document(request):
    serializer = DocumentSerializer(data=request.data)

    if serializer.is_valid():
        document = serializer.save()

        text = ingest_document(document)
        decision = decide_and_act(document, text)

        return Response({
            "message": "Document uploaded & processed successfully",
            "decision": decision,
        })

    return Response(serializer.errors, status=400)
