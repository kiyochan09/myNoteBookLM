"""Application-owned paths and portable file references; no working-directory dependency."""
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlparse

PATH_KEYS = frozenset(('original_path', 'source_folder', 'source_path', 'file_path', 'image_path'))
URL_PREFIXES = ('http:', 'https:', 'data:', 'blob:', 'mailto:')

@dataclass(frozen=True)
class AppPaths:
    root: Path
    source_roots: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).resolve())
        object.__setattr__(self, 'source_roots', tuple(Path(p).resolve() for p in self.source_roots))

    @property
    def data(self): return self.root / 'data'
    @property
    def media(self): return self.data / 'media'
    @property
    def uploads(self): return self.data / 'temp_uploads'
    @property
    def pdf_cache(self): return self.data / 'pdf_cache'
    @property
    def raw_pdfs(self): return self.data / 'raw_pdfs'
    @property
    def ocr_results(self): return self.data / 'ocr_results'
    @property
    def ocr_archives(self): return self.data / 'ocr_archives'
    @property
    def tcy_templates(self): return self.data / 'tcy_templates'
    @property
    def user_dictionary(self): return self.data / 'user_dictionary.json'
    @property
    def tcy_registry(self): return self.data / 'tcy_registry.json'
    @property
    def notebook_db(self): return self.data / 'my_notebook.db'
    @property
    def knowledge_db(self): return self.data / 'knowledge.db'
    @property
    def research_db(self): return self.data / 'research.db'
    @property
    def research_index_db(self): return self.data / 'research_index.db'
    @property
    def ai_settings(self): return self.data / 'research_ai.json'
    @property
    def token(self): return self.data / '.app_token'
    @property
    def models(self): return self.root / 'models'
    @property
    def ocr_runtime(self): return self.root / 'runtimes' / 'ocr'
    @property
    def research_runtime(self): return self.root / '.venv_research'
    @property
    def config_file(self): return self.root / 'config' / 'paths.json'
    @property
    def backup_log(self): return self.root / '管理' / 'バックアップ' / 'backup_retention_runs.jsonl'

    def research_model(self, revision):
        return self.models / 'research' / ('multilingual-e5-small-' + revision[:12])

    @property
    def backup_directory(self):
        config = self.config_file
        # Services with an injected temporary data root share the installed defaults.
        if not config.is_file() and self.root != APP_PATHS.root:
            config = APP_PATHS.config_file
        settings = json.loads(config.read_text(encoding='utf-8-sig'))
        value = settings.get('backup_directory')
        if not isinstance(value, str) or not value.strip():
            raise ValueError('config/paths.json のbackup_directoryを確認してください。')
        path = Path(os.path.expandvars(value)).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve()

    def resolve_file(self, value):
        """Resolve portable references at the point where a disk file is needed."""
        text = os.fspath(value)
        if text.lower().startswith(URL_PREFIXES):
            raise ValueError('URLをディスクのファイルとして使用できません。')
        if text.startswith('/media/'):
            path = (self.media / unquote(text[len('/media/'):])).resolve()
            if not path.is_relative_to(self.media.resolve()):
                raise ValueError('画像の保存場所を確認できません。')
            return path
        if text.lower().startswith('file:'):
            uri = urlparse(text)
            if uri.netloc not in ('', 'localhost'):
                text = '//' + uri.netloc + unquote(uri.path)
            else:
                text = unquote(uri.path)
                if re.match(r'^/[A-Za-z]:', text): text = text[1:]
        path = Path(text).expanduser()
        if path.drive and not path.is_absolute():
            raise ValueError('ドライブ相対パスは使用できません。')
        if path.is_absolute():
            resolved = path.resolve()
            for source in self.source_roots:
                if resolved.is_relative_to(source):
                    return self.root / resolved.relative_to(source)
            return resolved
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError('アプリの外を指す相対パスは使用できません。')
        return resolved

    def store_file(self, value):
        """Keep external locations/URLs; save application-owned files relative to root."""
        if value is None: return value
        text = os.fspath(value)
        if not text or text.startswith('/media/') or text.lower().startswith(URL_PREFIXES):
            return text
        resolved = self.resolve_file(text)
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root).as_posix()
        return text

    def media_file(self, value):
        path = self.resolve_file(value)
        if not path.is_relative_to(self.media.resolve()):
            raise ValueError('図版の保存フォルダー外は参照できません。')
        return path

    def media_url(self, value):
        if not value: return value
        text = os.fspath(value)
        try:
            path = self.media_file(text)
        except (ValueError, OSError):
            return text
        return '/media/' + quote(path.relative_to(self.media.resolve()).as_posix(), safe='/')

    def metadata(self, value):
        """Change known reference fields, leaving ordinary strings entirely intact."""
        original = value
        encoded = isinstance(value, str)
        if encoded:
            try: value = json.loads(value)
            except (ValueError, TypeError): return original
            if not isinstance(value, (dict, list)): return original
        def walk(item):
            if isinstance(item, dict):
                return {k: self.store_file(v) if k in PATH_KEYS and isinstance(v, str)
                        else walk(v) for k, v in item.items()}
            if isinstance(item, list): return [walk(v) for v in item]
            return item
        converted = walk(value)
        if encoded:
            return json.dumps(converted, ensure_ascii=False) if converted != value else original
        return converted

    def html(self, value):
        if not isinstance(value, str): return value
        pattern = r'(\b(?:src|href)\s*=\s*)(["\'])(.*?)\2'
        def attribute(match):
            original = match[3]
            converted = self.media_url(html.unescape(original))
            if converted == html.unescape(original): return match[0]
            return match[1] + match[2] + html.escape(converted, quote=True) + match[2]
        return re.sub(pattern, attribute, value, flags=re.I | re.S)

APP_PATHS = AppPaths(Path(__file__).resolve().parent.parent)
APP_ROOT = APP_PATHS.root
