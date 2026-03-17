#!/usr/bin/env python3
"""
兼容入口：复用主训练脚本，避免备用脚本与当前模型接口漂移。
"""

from graph_transform.scripts.train_graph_model import main


if __name__ == "__main__":
    main()
