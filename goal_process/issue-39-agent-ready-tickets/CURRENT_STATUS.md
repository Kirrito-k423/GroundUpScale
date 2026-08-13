# 当前状态

- **Goal：** issue-39-agent-ready-tickets
- **更新时间：** 2026-08-14T01:45:00+08:00
- **状态：** 绿
- **阶段：** INTEGRATE
- **验收进度：** 9/10 tickets（#41–#49 已 review PASS、集成、回归、推送并关闭）

## 一分钟摘要

- **目标：** 完成并集成 #41–#50。
- **已完成：** 固定 base/frontier/公共锁；#41–#46 完成双轴 review、语义集成、证据回放与关闭。
- **当前主阻塞：** #50 已由原生依赖解锁并以全新Agent/worktree启动。
- **关键证据：** `runs/20260814-C008/ITERATION.md`
- **下一步：** 执行、review、集成 #50，运行最终全套与冻结证据核验。
- **需要决策：** 无。

## 交付状态

- **代码：** `codex/integration-39` at `db06139`，已推送至 origin。
- **文档：** 本目录 WIP
- **费用报告：** 任务结束时生成 `RMB-Cost.md`（estimate，除非届时价格/汇率已验证）。

## NPU lock 基线

- **远端：** `root@192.168.9.225`，工作根 `/home/t00906153`。
- **Wrapper：** `/home/t00906153/.groundupscale/bin/with-ascend-lock`。
- **SHA-256：** `22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787`。
- **Device visibility：** `ASCEND_RT_VISIBLE_DEVICES=0`，逻辑设备 `npu:0`。
- **Smoke test：** owner 在持锁命令内存在、退出后清理；未初始化 NPU。

## 条件化 ETA

- **路径 A：** 若 tickets 以 fixture/已有证据为主且远端可用，预计多批并行完成，E1。
- **路径 B：** 若 required holdout 或兼容性阻塞，保留已完成提交并形成精确 evidence boundary，E1。
- **最晚决策点：** 发生确需用户授权或外部状态变化的不可推进阻塞时。
