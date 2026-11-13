# 从零开始学 AI 技术：给初学者的详细教程

> 面向完全没有 AI 基础、但会一点电脑操作的人。目标不是让你成为算法专家，而是让你在 3 个月内能独立做出可用的 AI 应用，并知道下一步往哪走。

## 0. 先搞清楚：你说的"做 AI"是哪一种

"做 AI"至少有三条路线，学习内容差别很大。先选一条主线，别同时学。

| 路线 | 做什么 | 需要的能力 | 适合的人 |
| --- | --- | --- | --- |
| 用 AI（应用层） | 调用大模型 API，做聊天机器人、知识库、自动化工具 | Python 基础、API、提示词、简单后端 | 大多数人，见效最快 |
| 做 AI（算法层） | 训练模型、读论文、改网络结构 | 数学、深度学习、PyTorch、算力 | 想进大厂算法岗/读研 |
| 建 AI 产品（工程层） | 把模型部署成稳定、可收费的产品 | 后端、数据库、运维、产品思维 | 想创业/做独立开发 |

**建议路线：先"用 AI"，再补"做 AI"的地基，最后按需要深入。** 理由：应用层反馈快，能立刻做出东西；底层知识可以在做的过程中逐步补，避免一上来被数学劝退。

下面按这个顺序讲。

---

## 1. 前置基础（2～4 周）

### 1.1 Python

AI 领域 90% 的代码是 Python。你不需要学成专家，但下面这些必须熟练：

- 变量、列表、字典、循环、条件
- 函数、参数、返回值
- 类与对象（会读即可，初期少写）
- 文件读写、异常处理
- `pip` 安装包、虚拟环境
- 列表推导式、`enumerate`、`zip`
- 会用 `requests` 发 HTTP 请求

检验标准：能独立写一个脚本，读取 CSV、统计某列平均值、把结果写到新文件。

推荐：
- 廖雪峰 Python 教程（中文，够用）
- Python 官方教程（英文，权威）
- 不要看视频看三个月，边写边查

### 1.2 数学（够用就行，不要死磕）

初期只需要"知道概念 + 会调用库"：

- 线性代数：向量、矩阵、矩阵乘法、转置
- 概率统计：均值、方差、条件概率、正态分布
- 微积分：导数、偏导、梯度（知道"往梯度反方向走能减小损失"即可）

不要一上来啃《线性代数应该这样学》。等你在 PyTorch 里遇到 `shape` 对不上、梯度爆炸时，再回头补对应章节，效率高十倍。

### 1.3 英语

AI 最新资料几乎全是英文。目标不是考托福，而是能读懂文档和报错。每天读 10 分钟官方文档即可。

### 1.4 命令行与 Git

```bash
# 必须会的命令
ls, cd, mkdir, cp, mv, rm
python --version
pip install 包名
git clone 仓库地址
git add . && git commit -m "说明"
git push
```

---

## 2. 搭好环境（半天）

### 2.1 安装 Python

- Windows：官网下载 Python 3.11+，安装时勾选 "Add Python to PATH"
- macOS：`brew install python@3.11`
- Linux：`sudo apt install python3 python3-pip python3-venv`

验证：

```bash
python3 --version
pip3 --version
```

### 2.2 虚拟环境（每个项目一个，必须养成习惯）

```bash
mkdir my-ai-project && cd my-ai-project
python3 -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
```

退出环境用 `deactivate`。

### 2.3 编辑器

- VS Code（免费、插件多）或 Cursor（AI 辅助写代码）
- 必装插件：Python、Jupyter、GitLens
- 学会用 Jupyter Notebook 做实验：适合一步步跑代码、看图、调试

### 2.4 常用工具一览

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

- `numpy`：数值计算
- `pandas`：表格数据处理
- `matplotlib`：画图
- `scikit-learn`：传统机器学习
- `jupyter`：交互式笔记本

### 2.5 没有显卡怎么办

- 初期完全够用：CPU + Google Colab 免费 GPU
- Colab：浏览器打开，新建 notebook，`运行时 → 更改运行时类型 → T4 GPU`
- 本地训练小模型，CPU 也能跑
- 不要一上来买显卡，等确定方向再说

---

## 3. 第一阶段：机器学习入门（3～4 周）

### 3.1 先建立直觉

机器学习 = 给程序一堆数据，让它自己找规律，而不是人写死规则。

核心概念（必须能用自己的话解释）：

- 特征（feature）与标签（label）
- 训练集 / 验证集 / 测试集
- 监督学习 vs 无监督学习
- 分类 vs 回归
- 过拟合（overfitting）与欠拟合
- 损失函数（loss）
- 评估指标：准确率、精确率、召回率、F1、MSE

### 3.2 用 scikit-learn 跑通第一个模型

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
X, y = load_iris(return_X_y=True)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 选模型并训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 预测与评估
pred = model.predict(X_test)
print("准确率:", accuracy_score(y_test, pred))
```

这段代码要能逐行讲清楚在干什么。跑通后，把它改成"用泰坦尼克号数据集预测乘客是否生还"。

### 3.3 这一阶段要做的项目

1. **鸢尾花分类**（跟着教程）
2. **泰坦尼克生还预测**（自己处理缺失值、特征编码）
3. **房价回归**（练习 MSE、特征缩放）
4. **手写数字识别**（用 scikit-learn 的 digits 数据集）

项目要求：
- 用 pandas 看过数据：有多少行、缺失值、分布
- 画至少 2 张图
- 记录不同模型的分数对比
- 写一份 200 字的结论：哪个模型好，为什么

### 3.4 推荐资源

- 吴恩达《Machine Learning Specialization》（Coursera，有中文字幕）
- scikit-learn 官方 User Guide
- Kaggle 的 "Titanic" 入门竞赛（有大量公开 notebook 可学）

---

## 4. 第二阶段：深度学习与 PyTorch（4～6 周）

### 4.1 先理解神经网络

- 神经元 = 加权求和 + 激活函数
- 层叠起来就是网络
- 训练 = 前向传播算损失 → 反向传播算梯度 → 更新参数
- 常见网络：全连接（MLP）、卷积（CNN）、循环（RNN）、Transformer

不需要手推公式，但要能画出"输入→网络→输出→损失→更新"的流程。

### 4.2 安装 PyTorch

去 pytorch.org 选择你的系统和是否用 GPU，复制安装命令。CPU 版示例：

```bash
pip install torch torchvision torchaudio
```

### 4.3 亲手写一个训练循环（MNIST）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    for images, labels in train_loader:
        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1}, loss {loss.item():.4f}")
```

重点理解这五行：

```python
pred = model(images)      # 前向传播
loss = loss_fn(pred, labels)  # 算误差
optimizer.zero_grad()     # 清空旧梯度
loss.backward()           # 反向传播
optimizer.step()          # 更新参数
```

几乎所有深度学习训练都是这个骨架，换的只是数据、模型和损失函数。

### 4.4 这一阶段的项目

1. **MNIST 手写数字**（上面的代码）
2. **CIFAR-10 图像分类**（加卷积层，观察准确率变化）
3. **电影评论情感分类**（文本，先做简单分词 + MLP）
4. **用自己的照片做一个二分类器**（猫/狗，体验数据收集和过拟合）

### 4.5 推荐资源

- 李宏毅《机器学习》（中文视频，讲得生动）
- fast.ai《Practical Deep Learning for Coders》（先做再懂）
- PyTorch 官方 60 分钟入门
- 《动手学深度学习》（李沐，免费在线，代码可跑）

---

## 5. 第三阶段：大模型（LLM）应用开发（4～8 周）

这是目前最容易做出产品、也最容易找到工作的方向。**不需要训练模型，重点是会用、会组合。**

### 5.1 先学会调用 API

绝大多数模型都兼容 OpenAI 风格接口。以 Python 为例：

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="你的API_KEY",
    base_url="https://api.example.com/v1",  # 换成你用的服务商地址
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一个简洁的中文助手。"},
        {"role": "user", "content": "用三句话解释什么是过拟合。"},
    ],
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

必须掌握：
- `messages` 的三种角色：system / user / assistant
- `temperature`：越低越稳定，越高越发散
- `max_tokens`：控制输出长度
- 流式输出（stream）：让回复逐字出现
- 错误处理：超时、限流、余额不足

### 5.2 提示词（Prompt）基本功

好提示词通常包含：

1. 角色：你是谁
2. 任务：要做什么
3. 输入：材料是什么
4. 约束：格式、长度、语气
5. 示例：给 1～2 个输入输出样例（few-shot）

模板：

```text
你是一名资深{角色}。
任务：{具体任务}
输入：{材料}
要求：
- 输出格式：{JSON/Markdown/表格}
- 长度：{限制}
- 不要编造输入中没有的信息
示例：
输入：...
输出：...
```

### 5.3 RAG：让模型回答你的私有资料

RAG（检索增强生成）= 先把文档切块、转成向量存起来；用户提问时，先检索最相关的几块，再连同问题一起发给模型。

流程：

```text
文档 → 切分 → Embedding → 向量数据库
                                  ↑
用户问题 → Embedding → 相似度检索 → 拼进 Prompt → LLM → 回答
```

最小实现思路：

1. 用 `pypdf` 读取 PDF
2. 按 500～1000 字切块，块之间留 100 字重叠
3. 调用 embedding 接口把每块转成向量
4. 存到 FAISS / Chroma / pgvector
5. 提问时检索 top 3～5 块，拼成上下文
6. 让模型"只根据上下文回答，不知道就说不知道"

项目：把你自己的笔记/PDF 做成一个能问答的知识库。

### 5.4 工具调用与 Agent

- Function Calling：让模型输出"我要调用哪个函数、参数是什么"，你的代码去执行
- Agent：模型自己决定多步操作（查资料→计算→再总结）
- 常用框架：LangChain、LlamaIndex、Dify、n8n（低代码）

建议：**先手写一遍最小的 RAG 和工具调用，再用框架。** 否则出了问题你不知道是框架的锅还是自己的锅。

### 5.5 这一阶段的项目

1. **命令行聊天机器人**（支持多轮对话、历史记录）
2. **个人知识库问答**（RAG，支持上传 PDF）
3. **网页摘要/翻译工具**（输入 URL，输出中文摘要）
4. **带工具的计算助手**（模型调用计算器、天气 API）
5. **自动写周报**（读取任务列表，生成结构化周报）

### 5.6 推荐资源

- OpenAI / Anthropic / DeepSeek 官方 API 文档（最值得读）
- LangChain 官方教程（选学，不要陷进去）
- Hugging Face 的 NLP Course（想深入时看）
- 《Building LLM Apps》类实战文章

---

## 6. 第四阶段：按方向深入（长期）

到这一步你已经能做出东西了，接下来按目标选一条：

### 6.1 想进算法岗

- 系统学 Transformer 原理，能手写 attention
- 读经典论文：Attention Is All You Need、BERT、GPT 系列、LoRA
- 微调实战：用 LoRA/QLoRA 微调一个 7B 模型
- 参加 Kaggle 或天池比赛
- 补数学：概率、优化、线代

### 6.2 想做 AI 产品/独立开发

- 后端：FastAPI + PostgreSQL + Redis
- 部署：Docker、云服务器、反向代理
- 推理服务：Ollama（本地）、vLLM（高并发）
- 监控与成本：token 计费、缓存、限流
- 评估：怎么判断你的 AI 功能"变好了"
- 产品：找一个具体场景，做小、做深、能收费

### 6.3 想研究多模态/图像/语音

- 图像：CNN → ViT → Diffusion（Stable Diffusion）
- 语音：Whisper（识别）、TTS（合成）
- 多模态：CLIP、视觉问答

### 6.4 微调入门（LoRA）

```text
基座模型（如 Qwen/Llama） + 少量领域数据 → LoRA 适配器 → 合并/加载 → 专用模型
```

关键点：
- 数据质量 > 数据数量
- 先试提示词和 RAG，不够再微调
- 微调不是"喂知识"，更适合"教风格/格式/任务"

---

## 7. 12 周学习计划（可直接照做）

| 周次 | 内容 | 产出 |
| --- | --- | --- |
| 第 1 周 | Python 基础 + 环境搭建 | 能跑通脚本、会建虚拟环境 |
| 第 2 周 | pandas / numpy / matplotlib | 完成一份数据探索报告 |
| 第 3 周 | 机器学习概念 + scikit-learn | 鸢尾花、泰坦尼克项目 |
| 第 4 周 | 模型评估与调参 | 写一份模型对比笔记 |
| 第 5 周 | 神经网络原理 | 能画出训练流程图 |
| 第 6 周 | PyTorch 基础 + MNIST | 训练准确率 > 97% |
| 第 7 周 | CNN + CIFAR-10 | 图像分类项目 |
| 第 8 周 | 调用 LLM API + 提示词 | 命令行聊天机器人 |
| 第 9 周 | Embedding + 向量检索 | 能按语义搜索笔记 |
| 第 10 周 | RAG 完整实现 | 个人知识库问答 |
| 第 11 周 | 工具调用 / 简单 Agent | 能查天气、算数的助手 |
| 第 12 周 | 整合 + 部署 | 一个能给别人用的作品 |

每天建议 1.5～2 小时：
- 30 分钟看文档/课程
- 60 分钟写代码
- 15 分钟记录踩坑和收获

---

## 8. 初学者最常见的 8 个坑

1. **教程地狱**：只看不写。每看 20 分钟必须动手写。
2. **数学恐惧**：一开始就想学完所有数学。用到再学。
3. **追新框架**：LangChain 换一版就重学。先掌握原理，框架只是工具。
4. **没有项目**：学完就忘。每个阶段必须有一个能演示的产出。
5. **数据不干净**：真实项目 70% 时间在洗数据，这是正常的。
6. **过拟合不自知**：训练集 99%、测试集 60%，说明模型在背答案。
7. **一上来微调大模型**：先试提示词和 RAG，成本低得多。
8. **忽视评估**：不知道"好"的标准，就无法改进。

---

## 9. 怎么判断自己学会了

不要用"看了多少课"衡量，用这些标准：

- 能不看教程，从零写出一个训练脚本
- 遇到报错能自己读堆栈、定位到文件和行号
- 能解释你的模型为什么有效、哪里可能失效
- 能估算一个 AI 功能的成本和延迟
- 有一个别人能打开使用的作品

达到前三条，算入门；达到第五条，算能干活。

---

## 10. 一份最小资源清单

- Python：廖雪峰教程 / 官方 Tutorial
- 机器学习：吴恩达 Coursera / scikit-learn 文档
- 深度学习：李宏毅视频 / 《动手学深度学习》
- PyTorch：官方 60 分钟入门
- 大模型：各家官方 API 文档 + Hugging Face NLP Course
- 实战：Kaggle、GitHub 上的开源小项目
- 社区：Hugging Face、Reddit r/MachineLearning、各类中文 AI 群

---

## 最后：今天就能做的三件事

1. 装好 Python 和 VS Code，建一个虚拟环境。
2. 跑通本文第 3.2 节的鸢尾花代码。
3. 注册一个模型 API（国内可选 DeepSeek、通义、智谱等），跑通第 5.1 节的调用示例。

先做出来，再理解。AI 这门技术，动手的人永远比只看的人走得远。

<div align="right"><sub>作者：wuming ｜ 2026-11-13</sub></div>
