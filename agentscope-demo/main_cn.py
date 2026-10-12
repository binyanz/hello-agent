# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法
"""
import asyncio
import os
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel, OllamaChatModel
from agentscope.pipeline import MsgHub
from agentscope.formatter import DashScopeMultiAgentFormatter, OllamaMultiAgentFormatter
from agentscope.message import Msg

from prompt_cn import ChinesePrompts
from game_roles import GameRoles
from structured_output_cn import (
    DiscussionModelCN,
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN
)
from utils_cn import (
    check_winning_cn,
    majority_vote_cn,
    get_chinese_name,
    format_player_list,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND,
)


def create_model_and_formatter():
    """根据 LLM_BACKEND 环境变量选择远程 DashScope 或本地 Ollama 模型+格式化器"""
    backend = os.getenv("LLM_BACKEND", "remote")
    if backend == "local":
        model = OllamaChatModel(
            model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            enable_thinking=False,
        )
        formatter = OllamaMultiAgentFormatter()
    else:
        model = DashScopeChatModel(
            model_name=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
            api_key=os.environ["DASHSCOPE_API_KEY"],
            enable_thinking=True,
        )
        formatter = DashScopeMultiAgentFormatter()
    return model, formatter


# 本地模型（尤其是较小的模型）在同一个 agent 实例被反复要求调用
# structured_model 对应的 generate_response 工具时，容易在第二次及以后退化成
# 直接输出JSON文字而不是真正"调用工具"，导致 Pydantic 校验失败、反复重试、
# 控制台刷屏。调用前先提醒一次能明显减少（但不能完全消除）这种情况。
STRUCTURED_OUTPUT_REMINDER = (
    "请注意：这次必须调用 generate_response 工具返回结构化结果，"
    "不要直接输出文字或JSON文本。"
)


async def remind_structured_output(agents) -> None:
    """在调用 structured_model 之前，提醒agent（或agent列表）必须走工具调用"""
    if isinstance(agents, ReActAgent):
        agents = [agents]
    reminder = Msg(name="系统", content=STRUCTURED_OUTPUT_REMINDER, role="system")
    for agent in agents:
        await agent.observe(reminder)


async def safe_call(agent, structured_model, msg=None):
    """单个 agent 的 structured_model 调用加 try/except，同样的 agentscope
    1.0.2 内部 bug（见 safe_fanout）在非 fanout 的单独调用点一样会触发，
    不能只保护 fanout 那两处。失败时返回 None，交给上层已有的兜底逻辑。"""
    try:
        if msg is not None:
            return await agent(msg, structured_model=structured_model)
        return await agent(structured_model=structured_model)
    except Exception as e:
        print(f"⚠️ {agent.name} 调用出错，视为本次无效：{e}")
        return None


async def safe_sequential(agents, msg=None):
    """agentscope.pipeline.sequential_pipeline 的等价实现，同样加 try/except：
    这个 agentscope 1.0.2 的内部 bug 不是只在 structured_model 校验失败时才
    触发，任何 ReActAgent 的 reply() 走到对应代码路径都可能炸，白天自由发言
    这种不带 structured_model 的调用一样会中招。单个 agent 失败时跳过它，
    把上一个成功的 msg 继续传给下一个 agent，避免整条链路/整局游戏崩掉。"""
    for agent in agents:
        try:
            msg = await agent(msg)
        except Exception as e:
            print(f"⚠️ {agent.name} 发言出错，跳过：{e}")
    return msg


async def safe_fanout(agents, msg, structured_model):
    """agentscope.pipeline.fanout_pipeline(enable_gather=False) 的等价实现，
    但给每个 agent 的调用单独加了 try/except：agentscope 1.0.2 在本地模型触发
    某些提前终止路径时，内部会出现 `chunk.metadata` 为 None 然后
    `.get(...)` 报 AttributeError 的情况，官方的 fanout_pipeline 没有做隔离，
    一个 agent 炸了会直接终止整个投票、进而终止整局游戏。这里保证单个 agent
    的异常只影响它自己这一票（视为无效票，交给上层已有的 None 兜底逻辑）。"""
    from copy import deepcopy

    results = []
    for agent in agents:
        try:
            result = await agent(deepcopy(msg), structured_model=structured_model)
        except Exception as e:
            print(f"⚠️ {agent.name} 调用出错，视为本轮无效：{e}")
            result = None
        results.append(result)
    return results


class ThreeKingdomsWerewolfGame:
    """三国狼人杀游戏主类"""

    def __init__(self):
        self.players: Dict[str, ReActAgent] = {}
        self.roles: Dict[str, str] = {}
        self.moderator = GameModerator()
        self.alive_players: List[ReActAgent] = []
        self.werewolves: List[ReActAgent] = []
        self.villagers: List[ReActAgent] = []
        self.seer: List[ReActAgent] = []
        self.witch: List[ReActAgent] = []
        self.hunter: List[ReActAgent] = []

        # 女巫道具状态
        self.witch_has_antidote = True
        self.witch_has_poison = True

    async def create_player(self, role: str, character: str) -> ReActAgent:
        """创建具有三国背景的玩家"""
        name = get_chinese_name(character)
        self.roles[name] = role

        model, formatter = create_model_and_formatter()
        # 本地模型 structured_model 调用的可靠性明显低于远程 qwen-max，
        # 调小 max_iters 让它更快放弃重试、交给上层已有的兜底逻辑处理，
        # 而不是刷满默认10轮重试日志。
        max_iters = 4 if os.getenv("LLM_BACKEND", "remote") == "local" else 10
        agent = ReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=model,
            formatter=formatter,
            max_iters=max_iters,
        )

        # 角色身份确认
        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )

        self.players[name] = agent
        return agent

    async def setup_game(self, player_count: int = 6):
        """设置游戏"""
        print("🎮 开始设置三国狼人杀游戏...")

        # 获取角色配置
        roles = GameRoles.get_standard_setup(player_count)
        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], player_count)

        # 创建玩家
        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)

            # 分配到对应阵营
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)

        # 游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )

        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")

    async def werewolf_phase(self, round_num: int):
        """狼人阶段"""
        if not self.werewolves:
            return None

        await self.moderator.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")

        # 狼人讨论
        async with MsgHub(
            self.werewolves,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"狼人们，请讨论今晚的击杀目标。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as werewolves_hub:
            # 讨论阶段
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await remind_structured_output(wolf)
                    await safe_call(wolf, DiscussionModelCN)

            # 投票击杀
            werewolves_hub.set_auto_broadcast(False)
            await remind_structured_output(self.werewolves)
            kill_votes = await safe_fanout(
                self.werewolves,
                await self.moderator.announce("请选择击杀目标"),
                structured_model=WerewolfKillModelCN,
            )

            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(kill_votes):
                # 检查vote_msg是否为None或metadata是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.werewolves[i].name] = vote_msg.metadata.get("target")
                else:
                    # 如果返回无效,随机选择一个目标
                    print(f"⚠️ {self.werewolves[i].name} 的击杀投票无效,随机选择目标")
                    import random
                    valid_targets = [p.name for p in self.alive_players if p.name not in [w.name for w in self.werewolves]]
                    votes[self.werewolves[i].name] = random.choice(valid_targets) if valid_targets else None

            killed_player, _ = majority_vote_cn(votes)
            return killed_player

    async def seer_phase(self):
        """预言家阶段"""
        if not self.seer:
            return

        seer_agent = self.seer[0]
        await self.moderator.announce("🔮 预言家请睁眼，选择要查验的玩家...")

        await remind_structured_output(seer_agent)
        check_result = await safe_call(seer_agent, get_seer_model_cn(self.alive_players))

        # 检查返回结果是否有效
        if check_result is None or not hasattr(check_result, 'metadata') or check_result.metadata is None:
            print(f"⚠️ 预言家查验失败,跳过此阶段")
            return

        target_name = check_result.metadata.get("target")
        if not target_name:
            print(f"⚠️ 预言家未选择查验目标,跳过此阶段")
            return

        target_role = self.roles.get(target_name, "村民")

        # 告知预言家结果
        result_msg = f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        await seer_agent.observe(await self.moderator.announce(result_msg))

    async def witch_phase(self, killed_player: str):
        """女巫阶段"""
        if not self.witch:
            return killed_player, None

        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")

        # 告知女巫死亡信息
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        await witch_agent.observe(await self.moderator.announce(death_info))

        # 女巫行动
        await remind_structured_output(witch_agent)
        witch_action = await safe_call(witch_agent, WitchActionModelCN)

        saved_player = None
        poisoned_player = None

        # 检查返回结果是否有效
        if witch_action is None or not hasattr(witch_action, 'metadata') or witch_action.metadata is None:
            print(f"⚠️ 女巫行动失败,视为不使用技能")
        else:
            if witch_action.metadata.get("use_antidote") and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用解药救了{killed_player}"))

            if witch_action.metadata.get("use_poison") and self.witch_has_poison:
                poisoned_player = witch_action.metadata.get("target_name")
                if poisoned_player:
                    self.witch_has_poison = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用毒药毒杀了{poisoned_player}"))

        # 确定最终死亡玩家
        final_killed = killed_player if not saved_player else None

        return final_killed, poisoned_player

    async def hunter_phase(self, shot_by_hunter: str):
        """猎人阶段"""
        if not self.hunter:
            return None

        hunter_agent = self.hunter[0]
        if hunter_agent.name == shot_by_hunter:
            await self.moderator.announce("🏹 猎人发动技能，可以带走一名玩家...")

            await remind_structured_output(hunter_agent)
            hunter_action = await safe_call(hunter_agent, get_hunter_model_cn(self.alive_players))

            # 检查返回结果是否有效
            if hunter_action is None or not hasattr(hunter_action, 'metadata') or hunter_action.metadata is None:
                print(f"⚠️ 猎人技能使用失败,视为放弃开枪")
                return None

            if hunter_action.metadata.get("shoot"):
                target = hunter_action.metadata.get("target")
                if target:
                    await self.moderator.announce(f"猎人{hunter_agent.name}开枪带走了{target}")
                    return target
                else:
                    print(f"⚠️ 猎人选择开枪但未指定目标,视为放弃")
                    return None

        return None

    def update_alive_players(self, dead_players: List[str]):
        """更新存活玩家列表"""
        for dead_name in dead_players:
            if dead_name:
                # 从存活列表移除
                self.alive_players = [p for p in self.alive_players if p.name != dead_name]
                # 从各阵营移除
                self.werewolves = [p for p in self.werewolves if p.name != dead_name]
                self.villagers = [p for p in self.villagers if p.name != dead_name]
                self.seer = [p for p in self.seer if p.name != dead_name]
                self.witch = [p for p in self.witch if p.name != dead_name]
                self.hunter = [p for p in self.hunter if p.name != dead_name]

    async def day_phase(self, round_num: int):
        """白天阶段"""
        await self.moderator.day_announcement(round_num)

        # 讨论阶段
        async with MsgHub(
            self.alive_players,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as all_hub:
            # 每人发言一轮
            await safe_sequential(self.alive_players)

            # 投票阶段
            all_hub.set_auto_broadcast(False)
            await remind_structured_output(self.alive_players)
            vote_msgs = await safe_fanout(
                self.alive_players,
                await self.moderator.announce("请投票选择要淘汰的玩家"),
                structured_model=get_vote_model_cn(self.alive_players),
            )

            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(vote_msgs):
                # 检查vote_msg是否为None或metadata是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.alive_players[i].name] = vote_msg.metadata.get("vote")
                else:
                    # 如果返回无效,默认弃票
                    print(f"⚠️ {self.alive_players[i].name} 的投票无效,视为弃票")
                    votes[self.alive_players[i].name] = None

            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)

            return voted_out

    async def run_game(self):
        """运行游戏主循环"""
        try:
            await self.setup_game()

            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")

                # 夜晚阶段
                await self.moderator.night_announcement(round_num)

                # 狼人击杀
                killed_player = await self.werewolf_phase(round_num)

                # 预言家查验
                await self.seer_phase()

                # 女巫行动
                final_killed, poisoned_player = await self.witch_phase(killed_player)

                # 更新死亡玩家
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)

                # 死亡公告
                await self.moderator.death_announcement(night_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                # 白天阶段
                voted_out = await self.day_phase(round_num)

                # 猎人技能
                hunter_shot = await self.hunter_phase(voted_out)

                # 更新死亡玩家
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                print(f"第{round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}")

        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    # 检查环境变量
    if os.getenv("LLM_BACKEND", "remote") != "local" and "DASHSCOPE_API_KEY" not in os.environ:
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY，或将 LLM_BACKEND 设为 local 使用本地 Ollama")
        return

    print("🎮 欢迎来到三国狼人杀！")

    # 创建并运行游戏
    game = ThreeKingdomsWerewolfGame()
    await game.run_game()


if __name__ == "__main__":
    asyncio.run(main())
