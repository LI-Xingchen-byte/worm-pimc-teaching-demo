# 本地整理与公开文件范围

公开仓库采用根目录白名单，规则在 [.gitignore](.gitignore)。
这里说明哪些内容由 Git 跟踪，以及哪些材料留在本地。

| 内容 | 用途 |
|---|---|
| `src/wormpimc/`、`pyproject.toml` | 库实现与安装配置 |
| `examples/` | 五个基础教学脚本和共享绘图辅助 |
| `docs/` | 使用、运行与验证专题 |
| `README.md`、`architecture.md`、`derivations.md` | 入口、实现结构和物理定义 |
| `tests/`、`configs/`、`validation/` | 回归检查、配置与独立参考 |
| `LICENSE`、`CONTRIBUTING.md`、`CHANGELOG.md` | 许可、协作与版本变化 |

## 保留在本地

- `local_archive/before_public_cleanup_20261005/`：整理前文档、配置和测试副本。
- `local_archive/advanced_examples/`：单粒子余弦势密度/谱专题及辅助程序。
- `experiments/`：原开发记录和验证原始数据，保留原路径供本地复查。
- `articles/`：个人阅读的参考 PDF。
- `tmp/`、`runs/`、`build/`、`dist/` 和缓存：运行或构建产物。

这些目录没有删除，默认也不会进入普通 Git 提交。
基础示例改名后，历史记录中的旧命令保持原文；当前命令以公开文档为准。

## 本地检查

首次发布前核对 Git 暂存区；今后增加文件时，也先预览待提交内容：

```shell
git add --dry-run .
git diff --cached --stat
```

`.gitignore` 控制新文件的默认跟踪范围，不审计文件内容，也不会移除已跟踪文件。
首次提交前仍需检查选中材料、个人信息和第三方代码来源。
新增公开的根目录文件或目录时，同时更新白名单及本说明。
