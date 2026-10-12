# AgentScope 三国狼人杀案例

对应 hello-agents 教程第六章 6.3 节，演示用 AgentScope 的消息驱动架构
（`MsgHub` + `fanout_pipeline`/`sequential_pipeline`）构建一个融合三国人物
的多智能体狼人杀游戏：每个玩家既是游戏角色（狼人/预言家/女巫/猎人/村民），
又叠加一个三国人物人格（刘备/曹操/诸葛亮...），通过 Pydantic 结构化输出
约束游戏规则（投票、查验、用药都必须是合法目标）。

## 文件说明

- `main_cn.py` — 游戏主控制器 `ThreeKingdomsWerewolfGame`：夜晚/白天阶段、
  胜负判定
- `prompt_cn.py` — 按游戏角色生成系统提示词
- `game_roles.py` — 角色配置表 + 三国人物性格描述
- `structured_output_cn.py` — 讨论/投票/查验/用药/击杀的 Pydantic 输出模型
- `utils_cn.py` — 投票统计、胜负判定、`GameModerator` 公告广播

## 运行

```bash
cd agentscope-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 见下面的远程/本地配置
python main_cn.py
```

## 远程 / 本地后端切换

跟 `autogen-demo` 同一个思路，`.env` 里 `LLM_BACKEND` 控制走 DashScope 远程还是
本地 Ollama，`create_model_and_formatter()`（`main_cn.py`）里切换的是
`DashScopeChatModel`+`DashScopeMultiAgentFormatter` 还是
`OllamaChatModel`+`OllamaMultiAgentFormatter`——agentscope 原生都支持，
`structured_model` 结构化输出约束两边都能用。

**远程**（`LLM_BACKEND=remote`，默认）：
```
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_MODEL=qwen-max
```
API Key 获取：https://dashscope.console.aliyun.com/apiKey

**本地**（`LLM_BACKEND=local`）：
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

本地模型人数多、轮次多的话会比较慢——6人局一整局下来可能是几十次 LLM 调用。

## 依赖注意事项

`requirements.txt` 里锁了两个版本，不是教程原文就有的，是实际 `pip install`
时踩出来的坑：
- `mcp==1.29.1`：`agentscope==1.0.2` 要求 `mcp>=1.13`，但默认会装最新的
  `mcp`（当前2.3.0），2.x 把 `streamablehttp_client` 改名成了
  `streamable_http_client`，导致 `import agentscope` 直接崩
- `tqdm`：`agentscope.evaluate` 内部用到，但 agentscope 自己没声明这个依赖
