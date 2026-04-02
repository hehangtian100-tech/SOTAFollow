# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个论文阅读整理仓库，用于分类管理 AI/机器人领域的 SOTA（State-of-the-Art）论文。

## 目录结构

```
├── RL/           # 强化学习 (Reinforcement Learning) 相关论文
│   └── 日报/      # RL 前沿每日日报
├── VLA/          # 视觉-语言-动作模型 (Vision-Language-Action) 相关论文
│   └── 日报/      # VLA 前沿每日日报
├── WorldModel/   # 世界模型 (World Model) 相关论文
│   └── 日报/      # WorldModel 前沿每日日报
└── contents.md    # 论文目录索引
```

## 使用方式

当用户提供论文进行阅读时：
1. 首先根据论文主题将其分类到对应目录
2. 如需保存论文摘要/笔记，使用对应目录下的 markdown 文件
3. 更新 contents.md 中的论文索引

## 日报功能

可通过 `/loop` 命令设置定时任务，定期推送各领域前沿日报（arxiv 最新论文、技术进展等）。

## 论文分类参考

- **RL**: Q-learning, Policy Gradient, RLHF, Transformer-based RL 等
- **VLA**: Robotics VLMs, RT-1/2, OpenVLA,具身智能等
- **WorldModel**: World Models, Dreamer, imagination-based planning 等
