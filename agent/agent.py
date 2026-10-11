import asyncio
import json
import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from agent.agent_loop import react_agent
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from agent.logger_config import get_logger
from pydantic import BaseModel, ConfigDict

logger = get_logger()

app = FastAPI()


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "Where is the capital of France?"}
        },
    )

    question: str
    chat_history: Optional[list] = None


class QueryResponse(BaseModel):
    answer: str


@app.post("/")
async def query(req: QueryRequest, raw_request: Request):
    """
    Research Agent API endpoint.
    Supports both JSON and SSE responses based on Accept header.
    When SSE is requested, sends periodic heartbeat comments to keep
    the connection alive while the agent is processing.
    """
    accept = raw_request.headers.get("accept", "")

    if "text/event-stream" in accept:
        # SSE mode: stream heartbeats while agent processes, then send answer
        async def sse_response():
            try:
                # Run agent in background task
                agent_task = asyncio.create_task(react_agent(req.question))

                # Send heartbeat comments every 5s to keep connection alive
                while not agent_task.done():
                    yield ": heartbeat\n\n"
                    try:
                        await asyncio.wait_for(asyncio.shield(agent_task), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue

                answer = await agent_task
            except Exception as e:
                logger.error("SSE error: %s", e)
                answer = f"Error: {e}"

            data = json.dumps({"answer": answer}, ensure_ascii=False)
            yield f"event: Message\ndata: {data}\n\n"

        return StreamingResponse(sse_response(), media_type="text/event-stream")

    # Normal JSON mode
    try:
        answer = await react_agent(req.question)
    except Exception as e:
        logger.error("Error: %s", e)
        answer = f"Error: {e}"
    return JSONResponse(content={"answer": answer})
