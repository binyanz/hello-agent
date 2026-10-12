import re

from llm_client import HelloAgentsLLM
from search_tool import ToolExecutor, search

REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。

示例一（History 为空，需要调用工具）:
Question: 现在北京天气怎么样？
History:
Thought: 我需要查询北京当前的天气，这不在我的知识库中，需要使用搜索工具。
Action: Search[北京今天天气]

示例二（History 中已经有搜索结果，即使搜索结果是多条、较长的网页摘要，只要能回答问题就不要再次搜索）:
Question: 今年苹果公司CEO是谁？
History:
Action: Search[苹果公司现任CEO]
Observation: [1] 苹果公司高管团队介绍 - 苹果官网
蒂姆·库克（Tim Cook）自2011年起担任苹果公司首席执行官（CEO），全面负责公司的全球运营。

[2] 库克谈苹果未来战略方向
在最近的采访中，蒂姆·库克表示苹果将继续加大在人工智能和可穿戴设备领域的投入...

[3] 苹果公司2026年第一季度财报电话会议纪要
首席执行官蒂姆·库克在电话会议上公布了最新季度的营收情况...
Thought: History 中的三条搜索结果都一致指向同一个事实——蒂姆·库克（Tim Cook）是苹果公司现任CEO，信息已经足够回答问题，不需要再次搜索。
Action: Finish[苹果公司现任CEO是蒂姆·库克（Tim Cook）。]

每次只输出一对 Thought 和 Action，不要自己编造 Observation 或后续轮次。生成 Action 前，必须先完整阅读 History 里已有的每一条搜索结果：只要能从中提炼出能回答问题的事实，哪怕搜索结果又长又杂，也要直接 Finish，绝对不要对同一个问题重复发起相同或相近的搜索。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 截断模型自己编造的后续 Thought/Action/Observation 轮次，只保留第一对
            truncate_match = re.search(
                r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
                response_text, re.DOTALL
            )
            if truncate_match:
                truncated = truncate_match.group(1).strip()
                if truncated != response_text.strip():
                    response_text = truncated
                    print("已截断多余的 Thought-Action 对")

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action, re.DOTALL).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


if __name__ == '__main__':
    llm_client = HelloAgentsLLM()

    tool_executor = ToolExecutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)

    agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor, max_steps=5)
    agent.run("华为最新发布的手机是什么型号？")
