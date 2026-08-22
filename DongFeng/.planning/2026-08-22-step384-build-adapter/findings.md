# Findings & Decisions

- 复用 STEP376 事务机制时必须私有加载，不能修改常规模块缓存中的旧 adapter，否则同进程测试会污染 STEP376 wiring。
- STEP384 锁：patcher SHA `2bdaf51e...d3820c`，candidate SHA `e352ac31...3b003`，reverse v4 SHA `2213dbae...4614b`。

## Requirements
-

## Research Findings
-

## Technical Decisions
| Decision | Rationale |
|----------|-----------|

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
-
