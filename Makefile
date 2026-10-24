# Polaris one-command reproduction pipeline (aligned with FlagshipAI Makefile practice)

.PHONY: help install test verify isa demo dispatch doctor clean

help:
	@echo "polaris 复现流水线:"
	@echo "  make install   # 建 venv + 安装（numpy/pytest + editable）"
	@echo "  make test      # 75 项测试全量"
	@echo "  make verify    # 测试 + ISA 1024 门禁 + 性质断言 390"
	@echo "  make isa       # ISA 统计 + 全量门禁"
	@echo "  make demo      # 端到端演示（含现场训练）"
	@echo "  make doctor    # 环境自检"
	@echo "  make clean     # 清理构建/模型产物"

install:
	python -m venv .venv
	.venv/Scripts/python.exe -m pip install --upgrade pip
	.venv/Scripts/python.exe -m pip install -e ".[dev]"

test:
	.venv/Scripts/python.exe -m pytest tests/ -q

verify: test
	.venv/Scripts/python.exe -m polaris.cli isa

isa:
	.venv/Scripts/python.exe -m polaris.cli isa

demo:
	.venv/Scripts/python.exe -m polaris.demo

dispatch:
	.venv/Scripts/python.exe -m polaris.cli dispatch "你好" --max-tokens 32

doctor:
	.venv/Scripts/python.exe -m polaris.cli doctor

clean:
	rm -rf .venv .polaris_model dist build *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
