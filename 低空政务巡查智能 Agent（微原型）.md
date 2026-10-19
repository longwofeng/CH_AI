# 低空政务巡查智能 Agent（微原型）

> **项目定位**：面向“无人机 + 测绘数据”的低空数字化场景，实现「航拍影像目标检测 → FastAPI 工具封装 → RAG 业务规则检索 → Agent 智能研判」的端到端闭环。
> **核心痛点**：传统无人机巡查只出图、出框，缺乏对测绘业务规则的理解；本项目让大模型作为“大脑”，调用视觉模型作为“眼睛”，结合本地规范作为“记忆”，自动化输出巡查研判报告。

---

## 🏗️ 一、 技术架构与仓库结构

### 架构链路
`用户上传航拍图` -> `Dify Agent 识别意图` -> `调用 HTTP 工具` -> `FastAPI (加载 YOLOv8 模型)` -> `返回检测 JSON` -> `Agent 检索本地测绘规范` -> `通义千问生成最终研判报告`

### GitHub 仓库结构规划
```text
/low-altitude-inspection-agent/
├── README.md                 # 核心介绍（即本文件）
├── requirements.txt          # 依赖清单 (ultralytics, fastapi, uvicorn, pillow)
├── Dockerfile                # 私有化部署打包文件
├── app/
│   ├── main.py               # FastAPI 核心代码（加载 YOLO，提供 /detect 接口）
│   └── yolo_weights/         # 存放 yolov8n.pt (可用 nano 版，速度快)
├── dify_agent/
│   ├── agent_prompt.txt      # 编排 Agent 的系统提示词
│   └── tool_schema.json      # 用于配置 Dify 自定义工具的 OpenAPI 格式
└── assets/
    ├── demo_input.jpg        # 测试用航拍图
    ├── detection_result.json # 接口返回示例
    └── screenshot_dify.png   # Dify 编排界面截图（明早补充）