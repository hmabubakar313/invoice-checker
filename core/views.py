from rest_framework.decorators import api_view
from rest_framework.response import Response
from core.models import AgentRun
from core.services.rag import rag_pipeline
from core.services.ingest import ingest_document
from core.services.orchestrator import decide_and_act, run_agent
from core.serializers import DocumentSerializer


@api_view(['POST'])
def ask_question(request):
    query = request.data.get("query")
    if not query or not isinstance(query, str):
        return Response({"query": ["This field is required and must be a string."]}, status=400)

    answer = rag_pipeline(query)
    return Response({"answer": answer})


@api_view(['POST'])
def upload_document(request):
    serializer = DocumentSerializer(data=request.data)

    if serializer.is_valid():
        document = serializer.save()

        text = ingest_document(document)
        decision = decide_and_act(document, text)
        state = run_agent(invoice_id=document.id, text=text)
        AgentRun.objects.create(
            document=document,
            state=state,
        )

        return Response({
            "message": "Document uploaded & processed successfully",
            "decision": decision,
        })

    return Response(serializer.errors, status=400)
