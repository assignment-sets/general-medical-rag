from modules.rag_pipeline import rag_chain
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
        result = rag_chain.invoke(cleaned_query)

        # Ensure JSON-serializable output
        if not isinstance(result, (dict, list)):
            result = {"response": str(result)}

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(e)}
        )
