import os
import threading
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

DEFAULT_LOCAL_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    支持两种后端，通过 LLM_BACKEND 环境变量或 backend 参数切换：
    - remote（默认）：调用任何兼容OpenAI接口的服务，流式响应，需要API key。
    - local：用 transformers 在本机加载开源模型推理，不需要网络和API key。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None,
                 timeout: int = None, backend: str = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.backend = backend or os.getenv("LLM_BACKEND", "remote")

        if self.backend == "local":
            self.model = model or os.getenv("LOCAL_MODEL_ID", DEFAULT_LOCAL_MODEL_ID)
            self._initLocalModel()
        else:
            self.model = model or os.getenv("LLM_MODEL_ID")
            apiKey = apiKey or os.getenv("LLM_API_KEY")
            baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
            timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

            if not all([self.model, apiKey, baseUrl]):
                raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

            self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def _initLocalModel(self):
        """
        加载本地开源模型和分词器，供 local 后端复用（只在第一次初始化时加载一次）。
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  正在加载本地模型 {self.model}（device={self._device}）...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._localModel = AutoModelForCausalLM.from_pretrained(self.model).to(self._device)
        print("✅ 本地模型加载完成。")

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        if self.backend == "local":
            return self._thinkLocal(messages, temperature)
        return self._thinkRemote(messages, temperature)

    def _thinkRemote(self, messages: List[Dict[str, str]], temperature: float) -> str:
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def _thinkLocal(self, messages: List[Dict[str, str]], temperature: float) -> str:
        from transformers import TextIteratorStreamer

        print(f"🧠 正在调用本地模型 {self.model}...")
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            modelInputs = self._tokenizer([text], return_tensors="pt").to(self._device)

            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            generationKwargs = dict(
                **modelInputs,
                max_new_tokens=512,
                do_sample=temperature > 0,
                streamer=streamer,
            )
            if generationKwargs["do_sample"]:
                generationKwargs["temperature"] = temperature

            thread = threading.Thread(target=self._localModel.generate, kwargs=generationKwargs)
            thread.start()

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for newText in streamer:
                print(newText, end="", flush=True)
                collected_content.append(newText)
            print()  # 在流式输出结束后换行
            thread.join()
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用本地模型时发生错误: {e}")
            return None


if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
