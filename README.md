# 碳动态模型可视化演示

这个项目是一个专门为平台优化的碳动态模型可视化应用，展示了9种不同的碳分解模型。

## 🚀 核心特性

- **云原生部署**: 专为Streamlit Cloud平台优化，无需本地服务器
- **现代化UI**: 响应式设计，支持各种设备访问
- **多种模型**: 包含9种不同的碳分解模型
- **交互式参数**: 实时调整模型参数并查看结果
- **数据可视化**: 图表和数据表格展示模拟结果

## 应用介绍

### 碳动态模型可视化 (app_carbon.py)
- 包含9种不同的碳分解模型（D1, D2, D3, D4, L1a, L2b, C1, RothC, DSSAT）
- 可调整各种模型参数进行模拟
- 实时展示碳储量动态变化曲线
- 提供模型方程和详细说明

## 🌐 Streamlit Cloud 部署

### 一键部署到 Streamlit Cloud

1. **Fork 此仓库**到你的GitHub账户

2. **访问 [Streamlit Cloud](https://share.streamlit.io/)**

3. **点击 "New app" 按钮**

4. **配置部署设置**：
   - Repository: 选择你fork的仓库
   - Branch: main
   - Main file path: `app_carbon.py`

5. **点击 "Deploy!"**

### 部署文件说明

- `Procfile`: 定义应用启动命令
- `requirements.txt`: 列出所有Python依赖
- `.streamlit/config.toml`: Streamlit配置
- `.streamlit/secrets.toml`: 密钥配置模板

## 📁 项目结构

```
carbon-and-greenhouse-gas-models/
├── app_carbon.py              # 主应用程序
├── carbon_models.py           # 碳模型实现
├── requirements.txt           # Python依赖
├── Procfile                   # 部署配置
├── README.md                  # 项目说明
└── .streamlit/
    ├── config.toml           # Streamlit配置
    └── secrets.toml          # 密钥配置模板
```

## 🔧 本地开发

如果你想在本地运行此应用：

1. **克隆仓库**
```bash
git clone <your-repo-url>
cd carbon-and-greenhouse-gas-models
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **运行应用**
```bash
streamlit run app_carbon.py
```

## 依赖说明

所有依赖项都列在 `requirements.txt` 文件中，包括：
- streamlit: 用于创建交互式Web应用
- numpy: 用于数值计算
- matplotlib: 用于数据可视化
- scipy: 用于科学计算
- pandas: 用于数据处理

## 使用说明

1. 在侧边栏选择想要演示的模型
2. 调整模型参数
3. 点击"运行模拟"按钮查看结果
4. 结果以图表和数据表形式展示
5. 可查看模型的核心方程和说明

## 📊 支持的模型

- **D1 Model**: 单池模型，最简单的碳分解模型
- **D2 Model**: 串联双池模型，碳从一个池流向另一个池
- **D3 Model**: 并联双池模型，碳同时分配到两个独立池
- **D4 Model**: 反馈模型，类似D2但参数不同
- **L1a Model**: 时变参数模型，分解率随时间变化
- **L2b Model**: 环境因子修正模型，考虑温湿度影响
- **C1 Model**: Gamma分布模型，描述分解率异质性
- **RothC Model**: 经典多池模型，5个不同分解特性的池
- **DSSAT-Century Model**: 区分植物残体组分和土壤有机碳

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！
