<div align="center">

# LLM-Ready

</div>



LLM-Ready 是一款推理服务器，可通过 **OpenAI 兼容 API** 在**本地**环境中促进使用开源大型语言模型 (LLM)，例如 FastChat、LLaMA 和 ChatGLM 。
LLM-Ready是基于[modelz-llm](https://github.com/tensorchord/modelz-llm.git),通过修改部分代码，可以支持在windows环境下运行。

## 功能

- **OpenAI 兼容 API**：LLM-Ready 为 LLM 提供了 OpenAI 兼容 API，这意味着您可以使用 OpenAI python SDK 或 LangChain 与模型交互。
- **自托管**：LLM-Ready 可以轻松部署在本地环境中。
- **开源 LLM**：LLM-Ready 支持开源 LLM，例如 FastChat、LLaMA 和 ChatGLM。

## 快速入门

### 安装

```bash
# or install from source
pip install git+https://github.com/generatemotion/llm-ready.git[gpu]
```

### 启动自托管API服务器

请首先按照以下步骤启动自托管 API 服务器：

```bash
python ready/cli.py -m bigscience/bloomz-560m --device cpu
```

目前, llm-ready支持如下的模型:

| 模型名称 | Huggingface 模型 | Docker 镜像 | 建议的GPU
| ---------- | ----------- | ---------------- | -- |
| FastChat T5 | `lmsys/fastchat-t5-3b-v1.0` | [modelzai/llm-fastchat-t5-3b](https://hub.docker.com/repository/docker/modelzai/llm-fastchat-t5-3b/general) | Nvidia L4(24GB) |
| Vicuna 7B Delta V1.1  | `lmsys/vicuna-7b-delta-v1.1` | [modelzai/llm-vicuna-7b](https://hub.docker.com/repository/docker/modelzai/llm-vicuna-7b/general) | Nvidia A100(40GB) |
| LLaMA 7B    | `decapoda-research/llama-7b-hf` | [modelzai/llm-llama-7b](https://hub.docker.com/repository/docker/modelzai/llm-llama-7b/general) | Nvidia A100(40GB) |
| ChatGLM 6B INT4    | `THUDM/chatglm-6b-int4` | [modelzai/llm-chatglm-6b-int4](https://hub.docker.com/repository/docker/modelzai/llm-chatglm-6b-int4/general) | Nvidia T4(16GB) |
| ChatGLM 6B  | `THUDM/chatglm-6b` | [modelzai/llm-chatglm-6b](https://hub.docker.com/repository/docker/modelzai/llm-chatglm-6b/general) | Nvidia L4(24GB) |
| Bloomz 560M | `bigscience/bloomz-560m` | [modelzai/llm-bloomz-560m](https://hub.docker.com/repository/docker/modelzai/llm-bloomz-560m/general) | CPU |
| Bloomz 1.7B | `bigscience/bloomz-1b7` | | CPU |
| Bloomz 3B | `bigscience/bloomz-3b` |  | Nvidia L4(24GB) |
| Bloomz 7.1B | `bigscience/bloomz-7b1` | | Nvidia A100(40GB) |

### 使用OpenAI python SDK

然后您可以使用 OpenAI python SDK 与模型交互：

```python
import openai
openai.api_base="http://localhost:8000"
openai.api_key="any"

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="any", messages=[{"role": "user", "content": "Hello world"}])
```

### 与Langchain整合

您还可以将 llm-ready 与 langchain 集成：

```python
import openai
openai.api_base="http://localhost:8000"
openai.api_key="any"

from langchain.llms import OpenAI

llm = OpenAI()

llm.generate(prompts=["Could you please recommend some movies?"])
```

## 支持的APIs

LLM-Ready 支持以下 API 来与开源大型语言模型交互：

- `/completions`
- `/chat/completions`
- `/embeddings`
- `/engines/<any>/embeddings`
- `/v1/completions`
- `/v1/chat/completions`
- `/v1/embeddings`

## 下一步
- 升级API支持OpenAPI新版本

## 感谢
- [modelz-llm](https://github.com/tensorchord/modelz-llm.git)
