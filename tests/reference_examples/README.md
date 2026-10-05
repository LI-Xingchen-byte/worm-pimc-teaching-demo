# 参考采样脚本

这些脚本原位于 `examples/01`–`04`，现用于回归检查，不作为基础教程入口。
从仓库根目录、安装库后可单独运行，例如：

```shell
python tests/reference_examples/free_brownian_bridge.py --samples 1000 --links 8
python tests/reference_examples/closed_worldline.py --samples 1000
python tests/reference_examples/closed_sampler.py
python tests/reference_examples/worm_sampler.py --verify-local-delta
```

后两个命令保存运行材料；closed sampler 只采样固定拓扑参考系综。
基础用法、更新展示和外势示例见 [README](../../README.md)。
