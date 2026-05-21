import os
import json
import logging
from scienceworld import ScienceWorldEnv
from openai import OpenAI
import faiss
import numpy as np

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RAGLongTermMemory:
    """非结构化长期记忆模块（基于简单的 FAISS 向量检索模拟）"""

    def __init__(self):
        # 模拟 128 维的嵌入空间
        self.index = faiss.IndexFlatL2(128)
        self.memory_store = []

    def add_memory(self, experience_text, is_success=True):
        status = "SUCCESS" if is_success else "MISTAKE"
        record = f"[{status}] {experience_text}"
        self.memory_store.append(record)
        # 模拟生成 Embedding 向量
        fake_vector = np.random.rand(1, 128).astype('float32')
        self.index.add(fake_vector)
        logging.info(f"长期记忆已沉淀: {record}")

    def retrieve(self, current_context, k=2):
        if not self.memory_store:
            return "No historical memories yet."
        # 模拟检索最相关的过往经验
        fake_query = np.random.rand(1, 128).astype('float32')
        D, I = self.index.search(fake_query, min(k, len(self.memory_store)))
        retrieved_items = [self.memory_store[idx] for idx in I[0] if idx < len(self.memory_store)]
        return "\n".join(retrieved_items)


class KnowledgeEnhancedAgent:
    """具备双循环记忆能力的文本交互智能体"""

    def __init__(self, config):
        # 💡 安全脱敏检测：优先从环境变量中获取 API KEY
        self.api_key = os.environ.get("OPENAI_API_KEY") or config.get("api_key")
        if not self.api_key or "YOUR_" in self.api_key:
            raise ValueError("[安全警示] 未检测到有效的 OPENAI_API_KEY，请设置环境变量进行脱敏加载！")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.working_memory = []  # 短期工作记忆（时序轨迹）
        self.long_term_memory = RAGLongTermMemory()

        # 预置一些跨任务泛化常识
        self.long_term_memory.add_memory("To melt an ice cube, you must place it inside a stove and turn the stove on.",
                                         is_success=True)
        self.long_term_memory.add_memory("Do not attempt to pick up fixed environments like stoves or tables.",
                                         is_success=False)

    def choose_action(self, observation, valid_actions, task_description):
        # 1. 提取长期记忆
        retrieved_mem = self.long_term_memory.retrieve(observation)

        # 2. 构造短期轨迹线索
        history_str = "\n".join(self.working_memory[-5:]) if self.working_memory else "None"

        # 3. 构造 ReAct Prompt
        system_prompt = (
            "You are an advanced LLM Agent in ScienceWorld.\n"
            f"Task: {task_description}\n\n"
            f"[RETRIEVED LONG-TERM KNOWLEDGE]\n{retrieved_mem}\n\n"
            f"[SHORT-TERM TRAJECTORY]\n{history_str}\n\n"
            f"Current Observation: {observation}\n"
            f"Available Actions: {', '.join(valid_actions)}\n\n"
            "Respond strictly in format:\nThought: <reasoning>\nAction: <exact valid action>"
        )

        # 4. 调用大模型
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.1
            )
            output = response.choices[0].message.content

            # 解析 Action
            action = "look around"  # 默认兜底动作
            for line in output.split("\n"):
                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
            return action
        except Exception as e:
            logging.error(f"LLM API 调用失败: {e}")
            return valid_actions[0] if valid_actions else "look around"

    def update_memory(self, action, observation, is_error=False):
        # 更新短期工作记忆
        self.working_memory.append(f"Action: {action} -> Obs: {observation}")
        # 如果触发错误恢复机制
        if is_error:
            self.long_term_memory.add_memory(f"Action '{action}' failed in context: {observation[:30]}",
                                             is_success=False)


def main():
    # 初始化脱敏配置
    config = {"model_name": "gpt-3.5-turbo", "api_key": os.environ.get("OPENAI_API_KEY", "placeholder")}

    try:
        agent = KnowledgeEnhancedAgent(config)
    except ValueError as e:
        print(e)
        return

    # 初始化 ScienceWorld 模拟环境
    env = ScienceWorldEnv()
    task_name = "change-of-state-melting"
    env.load(task_name, 0)

    task_description = env.getTaskDescription()
    obs, info = env.reset()

    print(f"=== 智能体启动成功，目标任务: {task_description} ===")

    max_steps = 30
    for step in range(max_steps):
        valid_actions = env.getValidActions()
        action = agent.choose_action(obs, valid_actions, task_description)

        print(f"第 {step + 1} 步 | 执行动作: {action}")
        next_obs, reward, done, info = env.step(action)

        # 异常与错误检测（判断是否需要错误恢复）
        is_error = "not possible" in next_obs.lower() or "nothing happens" in next_obs.lower()
        agent.update_memory(action, next_obs, is_error=is_error)

        obs = next_obs
        if done or reward >= 100:
            print(f"🎉 任务通关！最终得分: {reward}")
            break

    print("=== 模拟流结束 ===")


if __name__ == "__main__":
    main()