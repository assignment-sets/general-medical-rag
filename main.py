from modules.rag_pipeline.rag_engine_chain import rag_chain, query_classifier_chain
from fastapi.responses import JSONResponse
from fastapi import FastAPI, status, Query

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/rag/med/retrieve")
def retrieve_from_rag(
    query: str = Query(..., min_length=3, description="Medical query"),
):
    cleaned_query = query.strip()

    if not cleaned_query or len(cleaned_query.split()) < 2:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Query must be a meaningful medical question (at least two words)."
            },
        )

    try:
        classification = query_classifier_chain.invoke(cleaned_query)
        is_valid = classification.get("is_valid_query", "").strip().lower() == "true"

        if is_valid:
            response = rag_chain.invoke(cleaned_query)
            if not isinstance(response, (dict, list)):
                response = {"response": str(response)}
        else:
            response = {
                "response": "Sorry, I can only assist with valid medical questions. Please rephrase your query to be more medically relevant."
            }

        return JSONResponse(status_code=status.HTTP_200_OK, content=response)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(e)}
        )
