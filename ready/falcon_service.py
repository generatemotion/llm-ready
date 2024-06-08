import argparse
import multiprocessing as mp
import time

import falcon
import msgspec
from falcon.asgi import App, Request, Response
from huggingface_hub import snapshot_download
from llmspec import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    PromptCompletionRequest,
)

from freeport import FreePort

from emb import Emb
from log import logger
from model import LLM
from uds import Client, run_server


class Ping:
    async def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = "Modelz LLM service"


class ChatCompletions:
    def __init__(self, client: Client) -> None:
        self.client = client

    async def on_post(self, req: Request, resp: Response):
        buf = await req.stream.readall()
        try:
            chat_req = ChatCompletionRequest.from_bytes(buf=buf)
        except msgspec.ValidationError as err:
            logger.info(f"Failed to parse request: {err}")
            # return 400 otherwise the client will retry
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        comp = await self.client.request(chat_req)
        logger.info(comp)
        if isinstance(comp, Exception):
            resp.status = falcon.HTTP_500
            resp.data = ErrorResponse.from_validation_err(
                comp, "internal error"
            ).to_json()
            return
        resp.data = comp.to_json()


class Completions:
    def __init__(self, client: Client) -> None:
        self.client = client

    async def on_post(self, req: Request, resp: Response):
        buf = await req.stream.readall()
        try:
            prompt_req = PromptCompletionRequest.from_bytes(buf=buf)
        except msgspec.ValidationError as err:
            logger.info(f"Failed to parse request: {err}")
            # return 400 otherwise the client will retry
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        comp = await self.client.request(prompt_req)
        logger.info(comp)
        if isinstance(comp, Exception):
            resp.status = falcon.HTTP_500
            resp.data = ErrorResponse.from_validation_err(
                comp, "internal error"
            ).to_json()
            return
        resp.data = comp.to_json()


class Embeddings:
    def __init__(self, client: Client) -> None:
        self.client = client

    async def on_post(self, req: Request, resp: Response, engine: str = ""):
        if engine:
            logger.info("received emb req with engine: %s", engine)

        buf = await req.stream.readall()
        try:
            embedding_req = EmbeddingRequest.from_bytes(buf=buf)
        except msgspec.ValidationError as err:
            logger.info(f"Failed to parse request: {err}")
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        emb = await self.client.request(embedding_req)
        if isinstance(emb, Exception):
            resp.status = falcon.HTTP_500
            resp.data = ErrorResponse.from_validation_err(
                emb, "internal error"
            ).to_json()
            return
        resp.data = emb.to_json()

def build_falcon_app(args: argparse.Namespace):
    if args.dry_run:
        snapshot_download(repo_id=args.model)
        snapshot_download(repo_id=args.emb_model)
        return
    freeport = FreePort(start=4000, stop=6000)        
    model_port = freeport.port
    freeport = FreePort(start=4000, stop=6000)
    emb_port = freeport.port
    barrier = mp.get_context("spawn").Barrier(3)
    model_proc = run_server(
        model_port, barrier, LLM, model_name=args.model, device=args.device
    )
    emb_proc = run_server(
        emb_port, barrier, Emb, model_name=args.emb_model, device=args.device
    )
    barrier.wait()

    # wait to detect the proc exitcode
    time.sleep(1)
    if any(proc.exitcode is not None for proc in [model_proc, emb_proc]):
        raise RuntimeError("failed to start the service")

    llm_client = Client(model_port)
    completion = Completions(llm_client)
    chat_completion = ChatCompletions(llm_client)
    emb_client = Client(emb_port)
    embeddings = Embeddings(emb_client)

    app = App()
    app.add_route("/", Ping())
    app.add_route("/completions", completion)
    app.add_route("/chat/completions", chat_completion)
    app.add_route("/embeddings", embeddings)
    app.add_route("/engines/{engine}/embeddings", embeddings)

    # refer to https://platform.openai.com/docs/api-reference/chat
    # make it fully compatible with the current OpenAI API endpoints
    app.add_route("/v1/completions", completion)
    app.add_route("/v1/chat/completions", chat_completion)
    app.add_route("/v1/embeddings", embeddings)
    app.add_route("/v1/engines/{engine}/embeddings", embeddings)
    return app