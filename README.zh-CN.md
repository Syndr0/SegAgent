<div align="center">

# SegAgent

**对话式三维分割 · 类型化工具 · 有证据支撑的后续分析**

连接 Qwen3.5 与 VoxTell，用于医学影像分割、掩膜感知对话和轮廓质量控制的研究框架。

[![Status: Research](https://img.shields.io/badge/Status-Research-7c3aed?style=flat-square)](#项目状态)
[![License: MIT](https://img.shields.io/badge/License-MIT-2563eb?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)](segagent/pyproject.toml)
[![TypeScript 5.9](https://img.shields.io/badge/TypeScript-5.9-3178c6?style=flat-square&logo=typescript&logoColor=white)](segagent/frontend/package.json)

[![Planner: Qwen3.5-9B](https://img.shields.io/badge/Planner-Qwen3.5--9B-6366f1?style=flat-square)](docs/architecture.md#models-and-image-inputs)
[![Segmentation: VoxTell](https://img.shields.io/badge/Segmentation-VoxTell-0891b2?style=flat-square)](docs/architecture.md#models-and-image-inputs)
[![LangGraph](https://img.shields.io/badge/LangGraph-1c3c3c?style=flat-square&logo=langgraph&logoColor=white)](docs/architecture.md#conversation-and-execution)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](segagent/backend/api.py)
[![React 19](https://img.shields.io/badge/React-19-149eca?style=flat-square&logo=react&logoColor=white)](segagent/frontend/package.json)
[![NiiVue](https://img.shields.io/badge/NiiVue-3D_viewer-475569?style=flat-square)](docs/architecture.md#components)

[English](README.md) · [安装指南](docs/getting-started.md) · [架构说明](docs/architecture.md) · [配置参考](docs/configuration.md) · [贡献指南](CONTRIBUTING.md)

</div>

![SegAgent：持久化病例对话、Qwen3.5 规划、VoxTell 分割及按 ROI 角色分类的 QC](docs/assets/architecture.svg)

SegAgent 将规划与分割分开：**Qwen3.5-9B** 根据图像切片、病例对话和已有证据选择结构化动作；**VoxTell** 对完整三维影像进行分割。掩膜、叠加图、测量值、对话及运行证据与病例一起保存，支持继续提问和生成报告。

## 项目状态

应用位于 **[`segagent/`](segagent/)**，后端、前端、知识库和评估工具均在此目录。早期插件原型仅作为历史源码保存在 [`archive/prototype/`](archive/prototype/)。

> **研究软件。** 本项目未经临床验证，不是医疗器械。模型思考及模型间一致性不能作为临床证据；随附协议库是研究种子数据，不是获批指南。

## 主要能力

| 能力 | 实现 |
| --- | --- |
| 持久化多轮对话 | 同一病例保留消息历史，继续回答澄清问题，并引用已有病例内掩膜。 |
| 类型化规划 | 意图识别、确定性对话状态，以及协议查询、分割、检查、QC、视觉问答和最终回答等动作。 |
| 提示词归一化 | 区分解剖概念、病理结构与临床描述，校验后再发送给 VoxTell。 |
| 三维分割 | 对完整体数据推理，按有界提示词分块，保存掩膜、叠加图与测量值。 |
| 掩膜感知分析 | 调取已有掩膜，测量几何属性，结合叠加图继续问答或生成报告。 |
| 按 ROI 角色进行 QC | 区分器官、靶区、辅助结构及未解析标签，对适用器官进行模型对比并记录类型化指标证据。 |
| 实时界面 | React/NiiVue 查看器，增量显示模型思考、步骤与生成的掩膜。 |
| 工具接入 | MCP 工具与实验性 A2A 适配器复用同一组领域服务。 |

模型使用预训练权重，仓库没有训练或微调流程。

## 快速开始

需要 Python 3.11+、兼容的 PyTorch/torchvision、项目要求范围内的 Transformers 5.7+、Node.js 22.12+，并另外安装 VoxTell 与权重。先按 [完整安装指南](docs/getting-started.md) 准备环境。

```bash
git clone https://github.com/Syndr0/SegAgent.git
cd SegAgent/segagent
cp .env.example .env
```

示例配置已默认启用 Qwen3.5 和 thinking。填写真实模型路径：

```dotenv
SEGAGENT_PLANNER=qwen
SEGAGENT_LLM_MODEL=Qwen/Qwen3.5-9B
SEGAGENT_ENABLE_THINKING=true
SEGAGENT_VOXTELL_MODEL=/absolute/path/to/voxtell_v1.1
```

在应用目录启动后端：

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

另一个终端从仓库根目录启动前端：

```bash
cd segagent/frontend
npm install
npm run dev -- --host 127.0.0.1 --strictPort
```

打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)，上传 NIfTI 并发送消息。模型首次使用时加载；GPU 推理只启一个 API worker，不使用 `--reload`。

[配置参考](docs/configuration.md) 包括独立设备放置、图像预算、分块、存储和前端地址。[四张 3090 配置](segagent/.env.4x3090.example) 是一种布局示例，不是最低要求。`rule` 是用于开发和评估的规则规划基线；分割仍需 VoxTell。

## 对话工作流

1. 上传 `.nii` / `.nii.gz` 扫描，可附加同一网格的轮廓。
2. 发送请求，工作流根据意图和病例对话决定执行动作或询问缺失信息。
3. 检索协议、归一化提示词、调用 VoxTell、检查掩膜，或构建 QC 与报告证据。
4. 在同一病例继续询问已有掩膜，回答待澄清的问题，或上传编辑后的掩膜。
5. 结合测量值、产物、过程事件与局限检查最终结果。

明确命名的器官和特定病理结构可直接分割；按协议进行的 OAR 请求需要已提供或确认的治疗部位，明确点名的 OAR 可直接分割；泛指肿瘤时会询问空间信息，无法定位的尝试保持探索性标记。生成掩膜会直接出现在查看器中，没有强制批准/拒绝门。

QC 将模型结果视为参考。GTV、CTV、PTV 等临床靶区被测量和空间描述，并标记为 `not_automatically_assessed`，不会自动判定正确；辅助结构和未解析标签单独报告。

## 文档与源码

| 路径 | 内容 |
| --- | --- |
| [`segagent/`](segagent/) | SegAgent 应用：后端、前端、知识库、评估与测试 |
| [Architecture](docs/architecture.md) | 对话、模型、工具、QC、证据与接口 |
| [Configuration](docs/configuration.md) | 环境变量、默认值、设备与持久化路径 |
| [Getting started](docs/getting-started.md) | 依赖、权重和首次对话 |
| [Experiments](segagent/EXPERIMENTS.md) | 已实现评估与后续实验 |
| [Contributing](CONTRIBUTING.md) | 开发与提交约定 |
| [Security](SECURITY.md) | 数据处理与漏洞报告 |
| [`archive/prototype/`](archive/prototype/) | 历史插件原型，仅作源码存档 |

## 评估与研究边界

```bash
cd segagent
pytest -q
python evals/run_evals.py --dataset evals/golden_cases.jsonl
python evals/run_model_matrix.py --case-id case_... --thinking both
```

文本样例检查规则规划和检索；模型矩阵比较 Qwen3.5-9B/27B 的 thinking 开关，直接调用真实多模态规划器，不等同于端到端分割或临床评估。单元测试覆盖对话、门控、提示词、证据、QC、存储及运行时保护，不加载真实权重。指标与局限见 [评估说明](segagent/EXPERIMENTS.md)，本文不宣称实测基准结果。

应用面向本地、去标识 NIfTI 研究，没有身份验证、自动去标识或生产部署栈。证据检查和 ROI 角色判定不能保证事实或临床正确性。研究应用之外的使用仍需临床专家复核。历史原型中的 DICOM/RTSTRUCT 集成不是当前上传接口。

## 贡献、引用与许可

欢迎按 [贡献指南](CONTRIBUTING.md) 提交改进。引用时请记录 commit、模型 revision、配置和数据划分，并按上游指引引用 [VoxTell](https://github.com/MIC-DKFZ/VoxTell)、[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B) 等依赖。

仓库贡献采用 [MIT 许可证](LICENSE)，外部依赖、预训练权重及上游派生组件仍受各自许可证约束。
