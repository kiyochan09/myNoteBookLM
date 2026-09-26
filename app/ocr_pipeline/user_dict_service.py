# -*- coding: utf-8 -*-
"""
User Dictionary Service
ユーザーが登録した補正辞書（data/user_dictionary.json）を読み込み、
OCRテキストおよびWordエクスポートテキストに自動適用するエンジン。
旧字体・異体字の正規化、完全一致置換、正規表現置換を統合処理する。
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
USER_DICT_PATH = BASE_DIR / "data" / "user_dictionary.json"

class UserDictService:
    _instance = None
    _last_mtime = 0.0
    _exact_rules: List[Tuple[str, str]] = []
    _regex_rules: List[Tuple[re.Pattern, str]] = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.reload_if_needed()

    def reload_if_needed(self, force: bool = False):
        if not USER_DICT_PATH.exists():
            return

        try:
            mtime = USER_DICT_PATH.stat().st_mtime
            if not force and mtime <= self._last_mtime:
                return

            with open(USER_DICT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            exact_map = {}
            regex_list = []

            # 1. 'rules' 配列形式の読み込み (UIから送信される標準形式)
            rules = data.get("rules", [])
            for r in rules:
                if r.get("active") is False:
                    continue
                pat = r.get("pattern", "")
                rep = r.get("replacement", "")
                rtype = r.get("type", "exact")
                if not pat:
                    continue

                if rtype == "regex":
                    try:
                        compiled = re.compile(pat)
                        regex_list.append((compiled, rep))
                    except Exception:
                        pass
                else:
                    exact_map[pat] = rep

            # 2. 'exact_replacements' 辞書形式の読み込み (後方互換性)
            for k, v in data.get("exact_replacements", {}).items():
                if k and k not in exact_map:
                    exact_map[k] = v

            # 3. 'regex_replacements' 配列形式の読み込み
            for r in data.get("regex_replacements", []):
                if isinstance(r, dict):
                    pat = r.get("pattern", "")
                    rep = r.get("replacement", "")
                elif isinstance(r, (list, tuple)) and len(r) >= 2:
                    pat, rep = r[0], r[1]
                else:
                    continue
                if pat:
                    try:
                        compiled = re.compile(pat)
                        regex_list.append((compiled, rep))
                    except Exception:
                        pass

            # 長いパターンから順に適用して部分一致の誤爆を防ぐ
            sorted_exact = sorted(exact_map.items(), key=lambda x: len(x[0]), reverse=True)
            self._exact_rules = sorted_exact
            self._regex_rules = regex_list
            self._last_mtime = mtime

        except Exception as e:
            print(f"[UserDictService] Error loading user dictionary: {e}")

    def apply(self, text: str) -> str:
        """テキストにユーザー補正辞書ルールを適用する。"""
        if not text:
            return ""

        self.reload_if_needed()

        s = text

        # 1. 完全一致置換
        for pat, rep in self._exact_rules:
            if pat in s:
                s = s.replace(pat, rep)

        # 2. 正規表現置換
        for pattern, rep in self._regex_rules:
            try:
                s = pattern.sub(rep, s)
            except Exception:
                pass

        return s

def apply_user_dict(text: str) -> str:
    """ユーザー補正辞書を適用するヘルパー関数"""
    return UserDictService.get_instance().apply(text)
