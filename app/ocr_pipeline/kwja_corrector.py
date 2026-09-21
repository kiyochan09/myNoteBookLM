"""
KWJA (Kyoto University Web Japanese Analyzer) Adapter Module
京都大学日本語統合解析器 (KWJA) による形態素解析・Typo誤字脱字検出・文脈校正エンジン。
"""

import sys
import logging
import uuid
import difflib
from typing import List, Dict, Any, Optional

logger = logging.getLogger("kwja_corrector")

_KWJA_CLIENT = None
_KWJA_AVAILABLE: Optional[bool] = None


def is_kwja_available() -> bool:
    """KWJA が実行可能環境にインストールされているか確認"""
    global _KWJA_AVAILABLE
    if _KWJA_AVAILABLE is not None:
        return _KWJA_AVAILABLE
    try:
        import kwja
        _KWJA_AVAILABLE = True
    except ImportError:
        _KWJA_AVAILABLE = False
    return _KWJA_AVAILABLE


def get_kwja_client():
    """KWJA クライアントの遅延ロード"""
    global _KWJA_CLIENT
    if _KWJA_CLIENT is not None:
        return _KWJA_CLIENT

    if not is_kwja_available():
        return None

    try:
        from kwja.cli.client import TypoModule
        _KWJA_CLIENT = TypoModule()
        logger.info("KWJA TypoModule initialized successfully.")
    except Exception as e:
        logger.warning(f"Failed to initialize KWJA TypoModule: {e}")
        _KWJA_CLIENT = None

    return _KWJA_CLIENT


def extract_kwja_candidates(text: str, context_window: int = 30) -> List[Dict[str, Any]]:
    """
    KWJA を用いてテキストから誤字脱字・不自然な助詞等の校正候補を抽出する。
    KWJA 未導入環境では空リストを返し、安全にフォールバックする。
    """
    if not text or not text.strip():
        return []

    if not is_kwja_available():
        return []

    client = get_kwja_client()
    if not client:
        return []

    candidates: List[Dict[str, Any]] = []

    try:
        lines = text.splitlines()
        char_offset = 0

        for line in lines:
            if not line.strip():
                char_offset += len(line) + 1
                continue

            try:
                corrected_line = client.apply_typo(line)
                if corrected_line != line:
                    matcher = difflib.SequenceMatcher(None, line, corrected_line)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag in ("replace", "delete", "insert"):
                            orig_chunk = line[i1:i2]
                            sugg_chunk = corrected_line[j1:j2]
                            
                            start_pos = char_offset + i1
                            end_pos = char_offset + i2
                            
                            cb_start = max(0, start_pos - context_window)
                            cb = text[cb_start:start_pos]
                            ca_end = min(len(text), end_pos + context_window)
                            ca = text[end_pos:ca_end]

                            candidates.append({
                                "id": f"cand_kwja_{uuid.uuid4().hex[:8]}",
                                "original": orig_chunk,
                                "suggested": sugg_chunk,
                                "reason": "KWJA Typo/文脈補正",
                                "engine": "kwja",
                                "start_idx": start_pos,
                                "end_idx": end_pos,
                                "context_before": cb,
                                "context_after": ca,
                                "full_sentence": line,
                                "status": "pending"
                            })
            except Exception as inner_err:
                logger.debug(f"KWJA single line error: {inner_err}")

            char_offset += len(line) + 1

    except Exception as err:
        logger.warning(f"KWJA extraction error: {err}")

    return candidates


def correct_text_with_kwja(text: str) -> str:
    """KWJA でテキスト全体を一括校正"""
    if not text or not is_kwja_available():
        return text

    client = get_kwja_client()
    if not client:
        return text

    try:
        lines = text.splitlines()
        corrected_lines = []
        for line in lines:
            if not line.strip():
                corrected_lines.append(line)
            else:
                corrected_lines.append(client.apply_typo(line))
        return "\n".join(corrected_lines)
    except Exception as e:
        logger.warning(f"KWJA batch correction error: {e}")
        return text
