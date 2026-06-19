# MiniMind LLM / Agent 面试学习路线图

这份文档把作者在 README 中给出的 MiniMind 学习路线，改写成一套面向 **LLM / Agent 面试准备** 的学习方案。目标不是“跑通脚本”而已，而是能在面试中把每个模块讲清楚：它解决什么问题、核心公式是什么、代码在哪里、有哪些工程取舍、在 Agent 场景里如何串起来。

## 0. 学习目标：用 MiniMind 建立一张 LLM 全栈地图

MiniMind 的价值在于它用很小的模型和纯 PyTorch 实现，把大模型从零训练到 Agentic RL 的关键链路串成了可读代码。面试准备时建议把项目拆成 7 个层次：

| 层次 | 你要掌握的问题 | MiniMind 对应入口 |
|---|---|---|
| Tokenizer 与数据 | 文本如何变成 token？多轮消息、tool call、think 标签如何进入训练样本？ | `trainer/train_tokenizer.py`、`dataset/lm_dataset.py`、`model/tokenizer_config.json` |
| 模型结构 | Transformer、RoPE、GQA/KV cache、MoE、Tie Embedding 分别做什么？ | `model/model_minimind.py` |
| 预训练 | 为什么 next-token prediction 能学语言？loss 与数据质量如何影响能力上限？ | `trainer/train_pretrain.py` |
| SFT / 蒸馏 / LoRA | 如何把基座模型对齐成助手？如何注入格式、风格、工具调用与领域能力？ | `trainer/train_full_sft.py`、`trainer/train_distillation.py`、`trainer/train_lora.py`、`model/model_lora.py` |
| 偏好与强化学习 | DPO、PPO、GRPO、CISPO 的训练信号有什么差异？ | `trainer/train_dpo.py`、`trainer/train_ppo.py`、`trainer/train_grpo.py` |
| Tool Use / Agentic RL | 工具定义、工具调用、环境反馈、多轮 rollout、延迟奖励如何闭环？ | `scripts/eval_toolcall.py`、`trainer/train_agent.py`、`trainer/rollout_engine.py` |
| 推理与服务 | chat template、KV cache、OpenAI API 兼容服务、WebUI 如何落地？ | `eval_llm.py`、`scripts/serve_openai_api.py`、`scripts/web_demo.py` |

> 面试表达建议：不要只说“我会训练模型”。更好的说法是：“我用 MiniMind 复现过从 tokenizer、预训练、SFT、DPO/GRPO 到多轮 tool-use Agentic RL 的最小闭环，并能解释每个阶段的输入输出、loss、代码实现和工程限制。”

## 1. 作者路线图到面试能力的映射

README 的主路线是：**推理体验 → 数据准备 → Pretrain → SFT → 蒸馏 / LoRA / Tool Calling → DPO / PPO / GRPO / CISPO → Agentic RL → 部署与评测**。面试准备时，可以把它压缩成三条主线。

### 1.1 LLM 基础训练线

1. **Tokenizer**：理解 BPE / ByteLevel 的作用，以及为什么 `<tool_call>`、`<tool_response>`、`<think>` 需要作为模板标记稳定出现。
2. **Pretrain**：掌握 next-token prediction、本质是语言建模，能解释为什么它主要学习语言规律和事实分布。
3. **SFT**：掌握 instruction tuning、多轮对话模板、assistant-only loss mask、格式对齐与能力注入。
4. **Distillation**：区分黑盒蒸馏（学 teacher 输出）与白盒蒸馏（学 teacher token 分布）。
5. **LoRA**：掌握低秩增量矩阵、冻结基座权重、合并权重、适合垂域小数据微调的原因。

### 1.2 Alignment / RL 线

1. **DPO**：离线偏好学习，不需要 reward model 和在线 rollout；面试重点是 chosen/rejected 概率差与 `beta` 控制偏离程度。
2. **PPO**：在线策略优化，Actor 采样、Critic 估值、GAE 算优势、clip 防止更新过大、KL 约束防止跑偏。
3. **GRPO**：同一 prompt 采样一组回答，用组内 reward 均值/方差算相对优势，省掉 critic。
4. **CISPO**：可作为 GRPO loss 的小改动理解，重点是避免 ratio 被 clip 后梯度路径被硬截断。
5. **奖励设计**：区分 model-based、rule-based、environment-based reward；小模型特别要关注奖励稀疏和退化组问题。

### 1.3 Agent 工程线

1. **Tool Calling 数据格式**：system 挂工具列表，assistant 生成 tool_calls，tool 返回 observation，assistant 再继续回答。
2. **Chat Template**：把结构化消息展开成模型可学习的 token 序列，例如 `<tool_call>...</tool_call>` 和 `<tool_response>...</tool_response>`。
3. **Rollout**：策略模型基于上下文生成 action；若 action 是工具调用，则执行工具并把 observation 拼回上下文。
4. **延迟奖励**：Agentic RL 不只给单轮回答打分，而是对整条轨迹打分。
5. **训推分离**：训练侧负责 policy update，rollout / inference 侧负责高吞吐采样，二者通过轨迹和权重同步衔接。

## 2. 推荐 14 天学习计划

### Day 1：先跑通推理，建立整体感

- 阅读 README 的“项目介绍”“快速开始”“模型推理”。
- 跑 `eval_llm.py`，观察不同 `--weight` 对输出风格的影响。
- 面试复盘：解释为什么同一结构在 pretrain、full_sft、agent 权重下行为不同。

### Day 2：读数据和 tokenizer

- 重点读 `dataset/lm_dataset.py`：样本如何被编码、padding、mask、截断。
- 重点读 tokenizer 配置：特殊 token 和 chat template 如何影响训练/推理一致性。
- 面试复盘：解释“为什么训练和推理必须使用一致的 chat template”。

### Day 3-4：读模型结构

- 重点读 `model/model_minimind.py`：Embedding、RMSNorm、RoPE、Attention、MLP、MoE、LM Head。
- 画出一次 forward：`input_ids → hidden states → attention/mlp blocks → logits → CE loss`。
- 面试复盘：能讲清楚 KV cache 为什么能加速自回归推理，RoPE 为什么适合相对位置建模。

### Day 5：预训练

- 读 `trainer/train_pretrain.py`。
- 关注 dataloader、optimizer、学习率调度、梯度累积、DDP、checkpoint。
- 面试复盘：解释 pretrain loss 下降代表什么，以及为什么小模型的知识上限受参数量和数据质量限制。

### Day 6-7：SFT 与蒸馏

- 读 `trainer/train_full_sft.py` 和 `trainer/train_distillation.py`。
- 对比 SFT 的 CE loss 与蒸馏的 `CE + KL` 混合 loss。
- 面试复盘：解释 SFT、黑盒蒸馏、白盒蒸馏的区别和适用场景。

### Day 8：LoRA

- 读 `model/model_lora.py` 和 `trainer/train_lora.py`。
- 理解冻结原权重，只训练低秩 A/B 矩阵，以及合并权重的过程。
- 面试复盘：回答“为什么 LoRA 能省显存？什么时候不该只用 LoRA？”

### Day 9：Tool Calling 与自适应思考

- 读 README 中 Tool Calling / Adaptive Thinking 章节。
- 跑 `scripts/eval_toolcall.py`，观察工具调用、工具返回和最终回答的多轮上下文。
- 面试复盘：解释 `<think>` 开关是模板控制，不等价于单独训练一个 reasoning 模型。

### Day 10：DPO

- 读 `trainer/train_dpo.py`。
- 找到 actor/ref 模型、chosen/rejected logprob、DPO loss。
- 面试复盘：解释 DPO 为什么是 off-policy，以及它和 SFT 的差异。

### Day 11：PPO

- 读 `trainer/train_ppo.py`。
- 关注 rollout、reward model、critic、GAE、clip、KL。
- 面试复盘：解释 PPO 为什么工程复杂、显存更高、但适合在线探索。

### Day 12：GRPO / CISPO

- 读 `trainer/train_grpo.py`。
- 关注 group sampling、组内归一化优势、`loss_type=grpo/cispo`。
- 面试复盘：解释 GRPO 为什么可以不训练 critic，以及退化组为什么会让学习信号消失。

### Day 13：Agentic RL

- 读 `trainer/train_agent.py` 和 `trainer/rollout_engine.py`。
- 画出多轮轨迹：`prompt → tool_call → tool_response → answer → reward → update`。
- 面试复盘：解释 Agentic RL 与普通单轮 RLAIF 的区别：多轮、工具执行、环境反馈、延迟奖励。

### Day 14：部署、评测与项目表达

- 读 `scripts/serve_openai_api.py` 和 `scripts/web_demo.py`。
- 用 5 分钟准备项目介绍：背景、路线、你读过的核心代码、你能解释的算法、你做过的实验。
- 面试复盘：准备 3 个 trade-off：小模型限制、奖励稀疏、训练/推理模板一致性。

## 3. 高频面试题与回答框架

### 3.1 “请介绍一下 MiniMind 的模型结构”

回答框架：

1. 它是 decoder-only Transformer，自回归生成 token。
2. 输入先经过 token embedding，多层 Transformer block 后接 LM head 输出 vocabulary logits。
3. Attention 使用 RoPE 注入位置信息，推理时用 KV cache 复用历史 K/V。
4. MLP 可替换为 MoE：多个 expert + router，提升参数容量但只激活部分专家。
5. 训练目标通常是 next-token cross entropy。

### 3.2 “Pretrain 和 SFT 的区别是什么？”

回答框架：

- Pretrain 学语言和世界知识分布，数据通常是大量普通文本，目标是预测下一个 token。
- SFT 学指令跟随、对话格式和特定行为，数据是高质量 prompt/response 或多轮对话。
- MiniMind 的 SFT 还混入 tool call、reasoning、风格和知识蒸馏信号，因此不只是“聊天格式对齐”。

### 3.3 “DPO、PPO、GRPO 怎么比较？”

回答框架：

| 算法 | 数据来源 | 是否在线采样 | 是否需要 Critic | 优点 | 局限 |
|---|---|---:|---:|---|---|
| DPO | 静态偏好对 | 否 | 否 | 简单稳定、显存低 | 不能在线探索 |
| PPO | 当前策略 rollout | 是 | 是 | 经典、可在线优化 | 工程复杂、显存高、critic 难训 |
| GRPO | 当前策略分组 rollout | 是 | 否 | 省 critic、适合 LLM RL | 依赖组内 reward 差异，易退化组 |

### 3.4 “Agentic RL 和普通 RLHF/RLAIF 有什么不同？”

回答框架：

- 普通 RLAIF 常对单轮回答打分；Agentic RL 对一整条交互轨迹打分。
- Agent 中 action 可能是自然语言，也可能是结构化 tool call。
- tool call 会触发环境执行，observation 再进入上下文，模型继续规划。
- 奖励通常延迟到整轮结束才结算，更接近 environment-based reward。
- 工程上需要 rollout engine、工具执行器、格式解析、奖励聚合、权重同步。

### 3.5 “为什么小模型做 RL 容易失败？”

回答框架：

1. **能力边界低**：太难的任务全部答错，reward 全零或几乎无差异。
2. **奖励稀疏**：rule-based 0/1 奖励无法提供细粒度学习信号。
3. **退化组**：GRPO 中同组回答 reward 方差太小，优势接近 0。
4. **格式脆弱**：tool call / JSON / think 标签不闭合会导致环境无法解析。
5. **reward hacking**：模型可能学会利用奖励漏洞，而不是完成真实任务。

## 4. 读代码时建议画的 6 张图

1. **模型 forward 图**：token → embedding → N 层 block → logits → loss。
2. **预训练数据流图**：jsonl → dataset → input/label → CE loss。
3. **SFT mask 图**：哪些 token 算 loss，哪些 token 只是上下文。
4. **DPO 对比图**：prompt 下 chosen/rejected 的 logprob 差如何进入 loss。
5. **GRPO 分组图**：同一个 prompt 的多个 sample 如何计算组内优势。
6. **Agent rollout 图**：message history、tool call、tool response、reward 的循环。

## 5. 可交付的面试作品清单

如果你想把这个项目变成简历亮点，建议至少产出以下 4 个东西：

1. **一页结构图**：标出 MiniMind 的模型模块、训练阶段和推理入口。
2. **一份实验记录**：记录 pretrain/full_sft/agent 三类权重在同一 prompt 下的差异。
3. **一个自定义 LoRA**：用几十到几百条垂域数据训练一个可演示的小 adapter。
4. **一个轻 Agent demo**：定义 2-3 个工具，让模型完成“调用工具 → 读取结果 → 最终回答”的闭环。

## 6. 最终检查：你真的掌握了吗？

面试前用下面的问题自测。如果每题都能结合 MiniMind 代码讲 2-3 分钟，说明这个项目已经被你转化成了面试能力。

- Tokenizer 的特殊 token 和 chat template 为什么会影响 tool call 成败？
- Transformer block 中 attention、MLP/MoE、norm、residual 的顺序是什么？
- KV cache 缓存的是什么？为什么不是缓存 attention score？
- SFT 为什么通常只对 assistant 输出算 loss？
- 白盒蒸馏里的温度 `T` 和 KL 项分别起什么作用？
- LoRA 的低秩假设是什么？rank 太小或太大会怎样？
- DPO 的 reference model 为什么要固定？
- PPO 中 critic 估计不准会如何影响 actor？
- GRPO 为什么需要同一 prompt 采样多个回答？
- Agentic RL 中 reward 为什么是延迟的？tool execution 为什么不直接进入 loss，却会影响 policy update？

