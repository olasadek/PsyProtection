class QueryInput(BaseModel):
    medicine: str
    illness: str

@app.post("/generate-query/")
def generate_query(input: QueryInput):
    query = f"What are the best measurements for {input.medicine} addiction in {input.illness} patients?"
    return {"query": query}
