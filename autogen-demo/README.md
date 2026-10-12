# AutoGen 软件开发团队协作案例

对应 hello-agents 教程第六章 6.2 节，演示用 AutoGen (`0.7.4`) 的
`RoundRobinGroupChat` 编排四个角色（产品经理 → 工程师 → 代码审查员 → 用户代理）
轮流协作，自动化完成一个比特币价格 Streamlit 应用的开发。

## 文件说明

- `autogen_software_team.py` — 主案例代码，定义四个 `AssistantAgent` /
  `UserProxyAgent` 并用 `RoundRobinGroupChat` 编排协作流程
- `output_example.py` — 团队协作产出的参考示例（比特币价格 Streamlit 应用），
  来自教程仓库，不是本次运行生成的，仅作为预期产出的参照
- `requirements.txt` — 依赖（已修正教程原始依赖里的两处笔误：
  `asyncio` 是标准库不是 PyPI 包，`dotenv` 应为 `python-dotenv`）
- `.env.example` — 环境变量模板

## 运行

```bash
cd autogen-demo
python3.12 -m venv .venv   # 要求 Python >=3.10，系统默认 3.9 不行
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 然后按下面两种后端之一配置
python autogen_software_team.py
```

## 远程 / 本地后端切换

跟 `hello-agent/llm_client.py` 里 `HelloAgentsLLM` 的 `LLM_BACKEND` 开关一个思路，
`.env` 里 `LLM_BACKEND=remote`（默认）走 OpenAI 兼容远程 API，
`LLM_BACKEND=local` 走本地 Ollama：

```bash
brew install ollama
ollama serve &              # 启动本地服务
ollama pull qwen2.5:7b      # 拉取模型（跟 .env 里 OLLAMA_MODEL 对应）
```

`.env` 对应配置：
```
LLM_BACKEND=local
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

## 注意

非官方模型名（代理服务模型名、Ollama 本地模型名）SDK 无法自动推断能力，
两个客户端都显式传了 `model_info`（见 `autogen_software_team.py` 里的
`_MODEL_INFO`）。
