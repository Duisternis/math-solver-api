import asyncio
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from model import predict_latex
import io
import numpy
from pydantic import BaseModel
from fastapi import Request
import json
from math_solver import MathSolver
from sympy import latex

solver = MathSolver()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LatexInput(BaseModel):
    expression: str

@app.post("/predict/")
async def get_latex(file: UploadFile = File(...)):
    contents = await file.read()
    image = numpy.array(Image.open(io.BytesIO(contents)).convert("L"))
    
    latex_str = predict_latex(image)

    return JSONResponse(content={"latex": latex_str})

@app.post("/solve")
async def solve_latex(request: Request):
    # Read the request body
    body = await request.body()
    try:
        data = json.loads(body)
        expression = data.get('expression', '')
    except:
        expression = ""

    async def event_stream():
        yield "data: Received expression\n\n"

        parsed = solver.parse_latex(expression)
        if parsed is None:
            yield "data: Failed to parse the LaTeX expression.\n\n"
            return

        yield f"data: Parsed expression: {parsed}\n\n"

        solution = solver.handle_input(expression)
        print(solution)

        yield f"data: Final result: {solution}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")