# CAMEL 角色扮演案例：AI 心理学家 + AI 作家写书

对应 hello-agents 教程第六章 6.4 节，演示 CAMEL 的 `RolePlaying` —— 两个
智能体（"作家" 作为 AI User，负责提需求；"心理学家" 作为 AI Assistant，
负责给内容）通过"引导性提示"自主协作，合写一本关于拖延症心理学的科普电子书，
直到任一方输出 `<CAMEL_TASK_DONE>` 或达到 30 轮上限为止。

这份代码综合了 GitHub 仓库 `DigitalBookWriting.py` 和教程正文两个版本——
教程正文的版本多了 `with_task_specify=False`、空消息判空、以及带尖括号的
`<CAMEL_TASK_DONE>` 判定（更符合 CAMEL 实际的任务完成标记格式），这里采用
教程正文这个更完整的版本。

## 运行

```bash
cd camel-demo
python3.12 -m venv .venv   # 跟 agentscope-demo 一样，mcp 依赖链要求较新 Python
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 见下面的远程/本地配置
python DigitalBookWriting.py
```

脚本没有 `if __name__ == "__main__":` 保护（原仓库就是这样写的），顶层直接
执行，所以只能 `python DigitalBookWriting.py` 直接跑，不要 `import` 它。

## 远程 / 本地后端切换

跟其他几个 demo 同一个思路，`.env` 里 `LLM_BACKEND` 控制走远程还是本地：

**远程**（`LLM_BACKEND=remote`）：用 `ModelPlatformType.OPENAI_COMPATIBLE_MODEL`
对接任意 OpenAI 协议兼容服务（教程原文写的是 `ModelPlatformType.QWEN` 对接
阿里云百炼，但我们这边实际用的是 aihubmix 代理的非 Qwen 模型，所以换成了
更合适的 `OPENAI_COMPATIBLE_MODEL` 类型）：
```
LLM_MODEL_ID=your_model_id
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_base_url
```

**本地**（`LLM_BACKEND=local`）：CAMEL 原生支持 `ModelPlatformType.OLLAMA`
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b
```
```
LLM_BACKEND=local
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

## 依赖注意事项

`requirements.txt` 锁了 `mcp==1.29.1`，不是教程原文就有的——`camel-ai==0.2.75`
内部 `from mcp.server import FastMCP`，但默认装最新的 `mcp`（当前2.3.0）这个
导入路径不存在了，会直接报 `ImportError`。跟 `agentscope-demo` 踩的是同一类
坑（都是 mcp 包快速迭代导致的 API 搬迁），只是这次是反方向——agentscope 要
`mcp>=1.13`，camel-ai 要 `mcp<2.0`，锁定的版本刚好同时满足两边。
