# TCM-TargetTrial-RWE

**中医药真实世界证据的目标试验仿真框架**

## 核心创新点

1. **Target Trial Emulation** — 将 Hernán & Robins 目标试验范式应用于中医药疗效评价
2. **多因果估计器** — IPW / AIPW / TMLE 三类双重稳健估计
3. **重叠加权** — Li, Morgan & Zaslavsky (2018) 的 overlap weighting
4. **敏感性分析** — E-value + tipping point 未测量混杂评估
5. **预配置协议模板** — 4 个经典中药方剂的标准化试验协议

## 快速开始

```bash
pip install -e ".[dev]"
uvicorn backend.main:app --port 8011 --reload
```

## 预配置协议模板

| ID | 方剂 | 研究问题 | 结局指标 |
|----|------|---------|---------|
| `buzhong_yiqi_cancer_fatigue` | 补中益气汤 | 癌因性疲乏 | 疲乏评分变化 |
| `liuwei_dihuang_dn` | 六味地黄丸 | 糖尿病肾病进展 | eGFR 下降 |
| `danshen_aspirin_angina` | 丹参饮+阿司匹林 | 稳定型心绞痛 | 心绞痛频率 |
| `xiao_chaihu_hbv` | 小柴胡汤+NUC | 慢性乙型肝炎 | HBV DNA 抑制 |

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/protocols/` | GET/POST | 协议管理 |
| `/api/protocols/templates/list` | GET | 列出预配置模板 |
| `/api/protocols/templates/{id}` | GET | 获取模板详情 |
| `/api/protocols/templates/{id}/load` | POST | 加载模板为协议 |
| `/api/analysis/upload` | POST | 上传 EHR 数据 |
| `/api/analysis/propensity` | POST | 倾向评分分析 |
| `/api/analysis/causal` | POST | 因果估计 (IPW/AIPW/TMLE) |
| `/api/analysis/survival` | POST | 生存分析 |
| `/api/analysis/sensitivity` | POST | 敏感性分析 |
| `/api/analysis/balance` | POST | SMD 平衡诊断 |

## 因果估计方法

| 方法 | 特点 | 适用场景 |
|------|------|---------|
| IPW | 逆概率加权，可配置 bootstrap | 常规观察性研究 |
| AIPW | 双重稳健，GBM 结局模型 | 模型可能误设时 |
| TMLE | 目标最大似然，交叉拟合 | 高维数据，需高效估计 |
| Overlap | 重叠加权，自然截尾 | 极端倾向评分时 |

## 架构

```
backend/
  models/
    trial_protocol.py       试验协议数据结构
    protocol_templates.py   预配置协议模板
    causal_engine.py        IPW / AIPW / TMLE 估计器
    cost_effectiveness.py   成本效果分析
  analysis/
    propensity_score.py     倾向评分 + 重叠加权
    survival.py             KM / Cox / RMST
    sensitivity.py          E-value + tipping point
    target_trial.py         试验仿真编排器
  api/
    protocol.py             协议管理 + 模板 API
    analysis.py             分析端点
```

## 注意事项

本框架目前使用合成数据进行方法验证。要用于发表，需要接入真实中医院 HIS/EHR 数据。

## License

MIT
