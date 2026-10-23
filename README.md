# 低空政务巡查智能 Agent（微原型）

> **项目定位**：面向「无人机 + 测绘数据」的低空数字化场景，实现  
> **航拍影像目标检测 → FastAPI 工具封装 → RAG 业务规则检索 → Agent 智能研判** 的端到端闭环。

不同于传统「纯视觉识别」，本项目通过 Agent 机制，将**视觉感知（YOLOv8）**与**空间/测绘业务规范（RAG）**深度融合。无人机航拍图输入后，不仅输出物理目标的坐标与类别，更能由大模型结合巡查业务手册，生成结构化的研判建议。架构支持 Docker 私有化部署，适配政企内网环境。

---

## 🏗️ 技术架构

```
用户上传航拍图
    ↓
Dify Agent 识别意图
    ↓
调用 HTTP 工具（/detect_url）
    ↓
FastAPI 服务（加载 YOLOv8）
    ↓
返回检测 JSON（类别、数量、bbox、置信度）
    ↓
Agent 检索本地测绘规范（RAG）
    ↓
通义千问等大模型生成最终研判报告
```

### 核心能力

| 模块       | 技术              | 说明                                              |
|------------|-------------------|---------------------------------------------------|
| 视觉感知   | YOLOv8n           | 检测车辆、人员等低空巡查相关目标                  |
| 服务封装   | FastAPI           | 提供 `/detect_url`、`/health` 接口，支持 Dify 相对路径 |
| 智能研判   | Dify Agent + RAG  | 结合《低空无人机巡查规范》输出结构化报告          |
| 部署方式   | Docker            | 一键私有化，适合内网环境                          |

---

## 📁 仓库结构

```text
CH_AI/
├── README.md
├── requirements.txt
├── Dockerfile
├── main.py                                    # FastAPI + YOLOv8 推理服务
├── Agent 提示词草稿(agent_prompt).txt         # Dify Agent 系统提示词
├── 低空政务巡查智能 Agent（微原型）.md         # 项目说明
├── 低空无人机巡查规范（个人学习参考版）.md    # RAG 知识库文档（示例）
├── 影像数据/                                  # 测试影像
└── 20261019-234404.mp4                        # 演示视频
```

---

## 🚀 快速开始

### 1. 本地运行（推荐开发调试）

```bash
# 克隆仓库
git clone https://github.com/longwofeng/CH_AI.git
cd CH_AI

# 安装依赖
pip install -r requirements.txt

# 启动服务（默认端口 8000）
uvicorn main:app --host 0.0.0.0 --port 8000
```

服务启动后：

- 健康检查：`GET http://localhost:8000/health`
- 目标检测：`POST http://localhost:8000/detect_url`

**请求示例**：

```bash
curl -X POST http://localhost:8000/detect_url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/aerial.jpg"}'
```

**返回示例**：

```json
{
  "status": "success",
  "summary": {
    "car": 3,
    "person": 2
  },
  "details": [
    {
      "label": "car",
      "confidence": 0.87,
      "bbox": [120, 340, 280, 480]
    }
  ]
}
```

### 2. Docker 部署（推荐生产 / 内网）

```bash
# 构建镜像
docker build -t low-altitude-vision .

# 运行容器
docker run -d -p 8000:8000 \
  -e DIFY_BASE_URL=http://host.docker.internal:5001 \
  --name vision-service \
  low-altitude-vision
```

#### 常用环境变量

| 变量                    | 默认值                     | 说明                                           |
|-------------------------|----------------------------|------------------------------------------------|
| `YOLO_MODEL_PATH`       | `yolov8n.pt`               | 模型路径                                       |
| `DIFY_BASE_URL`         | `http://127.0.0.1:5001`    | Dify 文件服务地址（用于解析相对路径）          |
| `YOLO_IMAGE_SIZE`       | `1280`                     | 推理输入分辨率（航拍小目标建议较大）           |
| `CONFIDENCE_THRESHOLD`  | `0.25`                     | 置信度阈值                                     |
| `YOLO_IOU_THRESHOLD`    | `0.45`                     | NMS IoU 阈值                                   |
| `MAX_IMAGE_BYTES`       | `26214400`（25MB）         | 最大图片大小限制                               |
| `DOWNLOAD_TIMEOUT_SECONDS` | `30`                    | 图片下载超时时间（秒）                         |

---

## 🤖 Dify Agent 配置说明

1. **创建 Agent**，系统提示词参考仓库中的 `Agent 提示词草稿(agent_prompt).txt`。
2. **添加自定义工具**，指向本服务的 `/detect_url` 接口。
3. **挂载知识库**：将 `低空无人机巡查规范（个人学习参考版）.md` 上传到知识库。
4. **推荐工作流**：
   - 用户上传航拍图 → Agent 调用检测工具
   - 获取检测结果后，检索规范条款
   - 按固定格式输出报告：
     - 【巡查发现】：描述图像中的目标种类与数量
     - 【规范比对】：引用知识库中的具体条款
     - 【处置建议】：给出下一步人工核查建议

### Agent 提示词示例

```text
你是一个专业的“低空政务巡查智能助手”。
你的任务是协助测绘和城管人员分析无人机航拍影像。

工作流：
1. 用户会给你提供一张巡查图像，你必须调用 [目标检测工具] 分析图像中的目标和数量。
2. 工具会返回识别到的目标类别（如车、人等）和数量。
3. 你需要根据《低空无人机巡查测绘规范》（从知识库检索），对检测结果进行研判。
4. 输出格式要求：
   - 【巡查发现】：描述图像中的目标种类与数量。
   - 【规范比对】：引用知识库中的具体条款（如：超过了多少密度）。
   - 【处置建议】：给出下一步的人工核查建议（如：建议人工复核）。
```

---

## 🎯 检测目标类别

当前仅保留与低空政务巡查强相关的 COCO 类别，过滤无关目标（如 oven、refrigerator 等）：

- `car`
- `truck`
- `bus`
- `motorcycle`
- `bicycle`
- `person`

后续可根据实际业务扩展自定义训练类别（违建、水体异常、道路障碍等）。

---

## 📌 注意事项

- 本仓库中的《低空无人机巡查规范》为**个人学习参考版**，条款与阈值为示例设定，**不代表任何官方标准**。正式业务请以现行国家标准、行业标准及地方法规为准。
- 巡查成果仅作为辅助研判依据，涉及行政处罚或强制措施时，必须由具备执法资格的人员现场复核。
- 影像中的人脸、车牌等敏感信息在对外提供前应进行去标识化处理。
- 本项目仅供学习与研发测试使用。

---

## 📄 License

本项目仅供学习与研发测试使用。
```
