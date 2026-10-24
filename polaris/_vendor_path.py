"""Vendor path bootstrap.

北极星 POLARIS 的零重写铁律：世界顶级 AI 全栈 ifai（innovation-flagship-ai，
Apache-2.0，作者晨星）以**源码原样** vendor 于 ``third_party/ifai/``，
不做任何修改（含其 LICENSE / NOTICE）。本模块只做一件事：
把该目录注入 ``sys.path``，使 ``import ifai`` 可用。

为什么不重写、也不pip依赖第三方：
    ifai 是晨星本人同门旗舰（Apache-2.0 允许再分发），且未发布到 PyPI；
    vendor 源码保证「clone 即可跑、零网络、可复现」，
    与 requests/pip 等世界顶级项目对 urllib3/_vendor 的处理一致。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_IFAI = os.path.normpath(
    os.path.join(_HERE, "..", "third_party", "ifai")
)


def ensure_vendor_path() -> str:
    """把 vendor 的 ifai 源码根目录注入 sys.path（幂等）。"""
    if _VENDOR_IFAI not in sys.path:
        sys.path.insert(0, _VENDOR_IFAI)
    return _VENDOR_IFAI


__all__ = ["ensure_vendor_path"]
