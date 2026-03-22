#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desktop εφαρμογή συνομιλίας με Groq API.
- Χωρίς RAG
- Χωρίς Ollama
- Επιλογή μοντέλου από τη λίστα των ενεργών Groq models
- Υποστήριξη ελληνικών απαντήσεων
- Αναλυτικά logs σφαλμάτων/απαντήσεων
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "Groq Chat Desktop"
APP_VERSION = "1.5"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SYSTEM_PROMPT_EL = (
    "Απαντάς πάντα στα Ελληνικά με φυσική, σωστή και καθαρή διατύπωση. "
    "Κράτα τη μορφοποίηση ευανάγνωστη, με σύντομες παραγράφους όπου χρειάζεται. "
    "Αν ο χρήστης ζητήσει ρητά άλλη γλώσσα, ακολούθησε την οδηγία του."
)


@dataclass
class AppPaths:
    root: Path
    data_dir: Path
    logs_dir: Path
    config_path: Path
    runtime_log_path: Path


class RuntimeLogger:
    """Απλό logger που γράφει και στο UI και σε αρχείο."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level.upper()}] {message}"
        if details:
            try:
                serialized = json.dumps(details, ensure_ascii=False, indent=2)
            except Exception:
                serialized = repr(details)
            entry += f"\n{serialized}"
        entry += "\n"
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass
        return entry


class GroqAPIError(Exception):
    def __init__(self, message: str, *, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class GroqClient:
    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key.strip()
        self.timeout = timeout

    def chat_completion_with_retry(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
    ) -> Tuple[Dict[str, Any], int, Optional[int]]:
        requested = int(max_completion_tokens)
        payload = self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=requested,
        )
        return payload, requested, None

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise GroqAPIError("Δεν έχει οριστεί Groq API key.")

    def list_models(self) -> List[Dict[str, Any]]:
        self._ensure_api_key()
        url = f"{GROQ_BASE_URL}/models"
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise GroqAPIError(f"Αποτυχία σύνδεσης με Groq /models: {exc}") from exc

        if response.status_code != 200:
            raise GroqAPIError(
                "Αποτυχία ανάκτησης λίστας μοντέλων.",
                status_code=response.status_code,
                body=_safe_json_or_text(response),
            )

        payload = _safe_json_or_text(response)
        if not isinstance(payload, dict):
            raise GroqAPIError("Μη έγκυρη απάντηση από το /models.", body=payload)

        data = payload.get("data", [])
        if not isinstance(data, list):
            raise GroqAPIError("Το /models δεν επέστρεψε λίστα μοντέλων.", body=payload)
        return data

    def chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
    ) -> Dict[str, Any]:
        self._ensure_api_key()
        url = f"{GROQ_BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
        }

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise GroqAPIError(f"Αποτυχία σύνδεσης με Groq /chat/completions: {exc}") from exc

        parsed = _safe_json_or_text(response)
        if response.status_code != 200:
            raise GroqAPIError(
                "Η κλήση στο /chat/completions απέτυχε.",
                status_code=response.status_code,
                body=parsed,
            )

        if not isinstance(parsed, dict):
            raise GroqAPIError("Το /chat/completions επέστρεψε μη αναμενόμενο σώμα.", body=parsed)
        return parsed


def _safe_json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        try:
            return response.text
        except Exception:
            return "<no response body>"


def _extract_error_text(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg
        msg = body.get("message")
        if isinstance(msg, str):
            return msg
    if isinstance(body, str):
        return body
    return ""


def _extract_max_completion_limit(body: Any) -> Optional[int]:
    text = _extract_error_text(body)
    if not text:
        return None
    patterns = [
        r"less than or equal to [`']?(\d+)[`']?",
        r"maximum value .*? [`']?(\d+)[`']?",
        r"max(?:imum)? output tokens[^\d]*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
                if value > 0:
                    return value
            except Exception:
                pass
    return None


def _theme_palette(theme: str) -> Dict[str, str]:
    if theme == "dark":
        return {
            "hint": "#9eb0cc",
            "user_bg": "#14213a",
            "assistant_bg": "#0f1e34",
            "user_border": "#2f4f85",
            "assistant_border": "#2c5aa0",
            "user_title": "#dbe7fb",
            "assistant_title": "#dbe7fb",
            "text": "#f4f8ff",
            "code_bg": "#0f1728",
            "code_border": "#2c4777",
            "code_header_bg": "#16233a",
            "code_header_text": "#dce9ff",
            "code_text": "#eef4ff",
            "code_inline_bg": "#1a2741",
            "code_inline_border": "#36517f",
        }
    return {
        "hint": "#60748f",
        "user_bg": "#eef4ff",
        "assistant_bg": "#f7fbff",
        "user_border": "#b8ccf3",
        "assistant_border": "#cdddf6",
        "user_title": "#16345f",
        "assistant_title": "#21456f",
        "text": "#162538",
        "code_bg": "#eef4ff",
        "code_border": "#bfd1ef",
        "code_header_bg": "#dfeaff",
        "code_header_text": "#193d6b",
        "code_text": "#122033",
        "code_inline_bg": "#e6eefb",
        "code_inline_border": "#c8d7f0",
    }




def _render_inline_plain_html(text: str, palette: Dict[str, str]) -> str:
    code_span_style = (
        f"font-family: Consolas, 'Courier New', monospace; "
        f"font-size: 13px; background: {palette.get('code_inline_bg', '#e8eef8')}; "
        f"color: {palette.get('code_text', palette['text'])}; border: 1px solid {palette.get('code_inline_border', '#c6d4ec')}; "
        "border-radius: 6px; padding: 1px 6px;"
    )
    parts: List[str] = []
    last = 0
    for match in re.finditer(r"`([^`\n]+?)`", text):
        plain = text[last:match.start()]
        if plain:
            parts.append(html.escape(plain))
        code_text = html.escape(match.group(1))
        parts.append(f"<code style='{code_span_style}'>{code_text}</code>")
        last = match.end()
    tail = text[last:]
    if tail:
        parts.append(html.escape(tail))
    return "".join(parts).replace("\n", "<br>")


def _render_rich_text_html(text: str, palette: Dict[str, str], *, font_px: int = 14) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ""

    block_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    blocks: List[str] = []
    last = 0

    for match in block_pattern.finditer(text):
        plain = text[last:match.start()]
        if plain.strip():
            blocks.append(
                f"<div style='margin:0 0 12px 0; line-height:1.72; white-space:normal; word-wrap:break-word;'>"
                f"{_render_inline_plain_html(plain.strip(), palette)}"
                f"</div>"
            )

        lang = (match.group(1) or "").strip()
        code = (match.group(2) or "").rstrip("\n")
        lang_label = html.escape(lang) if lang else "Κώδικας"
        code_html = html.escape(code)
        blocks.append(
            f"<div style='margin:0 0 14px 0; border:1px solid {palette.get('code_border', '#bcd0ee')}; "
            f"border-radius:12px; background:{palette.get('code_bg', '#edf4ff')}; overflow:hidden;'>"
            f"<div style='padding:8px 12px; background:{palette.get('code_header_bg', '#ddeafe')}; "
            f"color:{palette.get('code_header_text', palette['user_title'])}; font-size:12px; font-weight:700; letter-spacing:0.3px;'>"
            f"{lang_label}</div>"
            f"<pre style='margin:0; padding:14px 16px; background:{palette.get('code_bg', '#edf4ff')}; "
            f"color:{palette.get('code_text', palette['text'])}; font-family: Consolas, 'Courier New', monospace; "
            f"font-size:13px; line-height:1.55; white-space:pre-wrap; word-break:break-word;'>"
            f"{code_html}</pre></div>"
        )
        last = match.end()

    tail = text[last:]
    if tail.strip() or not blocks:
        blocks.append(
            f"<div style='margin:0 0 4px 0; line-height:1.72; white-space:normal; word-wrap:break-word;'>"
            f"{_render_inline_plain_html(tail.strip() if tail.strip() else text.strip(), palette)}"
            f"</div>"
        )

    return (
        f"<div style='font-family: Segoe UI, Arial, sans-serif; font-size:{font_px}px; color:{palette['text']};'>"
        + "".join(blocks)
        + "</div>"
    )


def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks: List[Tuple[str, str]] = []
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text):
        lang = (match.group(1) or "").strip() or "text"
        code = (match.group(2) or "").rstrip("\n")
        if code.strip():
            blocks.append((lang, code))
    return blocks


def _extract_text_from_chat_payload(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GroqAPIError("Η απάντηση δεν περιέχει πεδίο 'choices'.", body=payload)

    first = choices[0]
    if not isinstance(first, dict):
        raise GroqAPIError("Το πρώτο choice δεν είναι έγκυρο αντικείμενο.", body=payload)

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        text = _normalize_content_to_text(content)
        if text.strip():
            return text.strip()

    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        text = _normalize_content_to_text(content)
        if text.strip():
            return text.strip()

    # Fallback σε πιο χαλαρό parsing
    for key in ("text", "response", "output_text"):
        value = first.get(key) or payload.get(key)
        text = _normalize_content_to_text(value)
        if text.strip():
            return text.strip()

    raise GroqAPIError("Δεν βρέθηκε αξιοποιήσιμο κείμενο στην απάντηση του Groq.", body=payload)


def _normalize_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif item.get("type") in {"text", "output_text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return str(content)


def mask_api_key(value: str) -> str:
    value = (value or "").strip()
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def guess_chat_model(model_id: str) -> bool:
    m = model_id.lower()
    excluded_fragments = [
        "whisper",
        "speech",
        "transcription",
        "translation",
        "orpheus",
        "guard",
        "prompt-guard",
        "moderation",
    ]
    return not any(fragment in m for fragment in excluded_fragments)


class ModelRefreshThread(QThread):
    success = Signal(list)
    error = Signal(str, dict)
    log = Signal(str)

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def run(self) -> None:
        try:
            client = GroqClient(self.api_key)
            self.log.emit("Ανάκτηση λίστας μοντέλων από Groq...")
            models = client.list_models()
            self.success.emit(models)
        except GroqAPIError as exc:
            details = {
                "status_code": exc.status_code,
                "body": exc.body,
                "traceback": traceback.format_exc(),
            }
            self.error.emit(str(exc), details)
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc), {"traceback": traceback.format_exc()})


class ChatThread(QThread):
    success = Signal(dict)
    error = Signal(str, dict)
    log = Signal(str)

    def __init__(
        self,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens

    def run(self) -> None:
        endpoint = f"{GROQ_BASE_URL}/chat/completions"
        try:
            client = GroqClient(self.api_key)
            request_preview = {
                "endpoint": endpoint,
                "model": self.model,
                "message_count": len(self.messages),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_completion_tokens": self.max_completion_tokens,
            }
            self.log.emit("Αποστολή αιτήματος στο Groq /chat/completions...")
            try:
                payload = client.chat_completion(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_completion_tokens=self.max_completion_tokens,
                )
                actual_tokens = self.max_completion_tokens
                auto_limit = None
            except GroqAPIError as exc:
                auto_limit = _extract_max_completion_limit(exc.body)
                if exc.status_code == 400 and auto_limit and auto_limit < int(self.max_completion_tokens):
                    self.log.emit(
                        f"Το μοντέλο δεν δέχτηκε max_completion_tokens={self.max_completion_tokens}. "
                        f"Νέα αυτόματη προσπάθεια με όριο {auto_limit}."
                    )
                    payload = client.chat_completion(
                        model=self.model,
                        messages=self.messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_completion_tokens=auto_limit,
                    )
                    actual_tokens = auto_limit
                else:
                    raise
            text = _extract_text_from_chat_payload(payload)
            request_preview["effective_max_completion_tokens"] = actual_tokens
            if auto_limit:
                request_preview["auto_adjusted_to_model_limit"] = auto_limit
            self.success.emit({"text": text, "raw": payload, "request": request_preview})
        except GroqAPIError as exc:
            details = {
                "endpoint": endpoint,
                "status_code": exc.status_code,
                "body": exc.body,
                "traceback": traceback.format_exc(),
                "parsed_max_completion_limit": _extract_max_completion_limit(exc.body),
            }
            self.error.emit(str(exc), details)
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc), {"endpoint": endpoint, "traceback": traceback.format_exc()})


class HeaderCard(QFrame):
    def __init__(self, title: str, subtitle: str):
        super().__init__()
        self.setObjectName("HeaderCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("HeroTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("HeroSubtitle")
        subtitle_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = self._build_paths()
        self.logger = RuntimeLogger(self.paths.runtime_log_path)

        self.models_raw: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, str]] = []
        self.last_answer_text: str = ""
        self.pending_assistant_text: str = ""
        self.last_code_blocks: List[Tuple[str, str]] = []
        self.model_refresh_thread: Optional[ModelRefreshThread] = None
        self.chat_thread: Optional[ChatThread] = None
        self.current_busy_scope: Optional[str] = None

        self.setWindowTitle(f"{APP_TITLE} v{APP_VERSION}")
        self.setMinimumSize(1200, 760)
        self.resize(1440, 920)
        self._set_safe_app_font()

        self.current_theme = "light"
        self._build_ui()
        self._load_settings()
        self._apply_theme()
        self._connect_signals()
        self._update_model_filter()
        self._log_info("Η εφαρμογή ξεκίνησε.")
        self.statusBar().showMessage("Έτοιμο.")

        if self.api_key_edit.text().strip():
            self.refresh_models(auto=True)

    def _build_paths(self) -> AppPaths:
        root = Path(__file__).resolve().parent
        data_dir = root / "data"
        logs_dir = data_dir / "logs"
        config_path = data_dir / "groq_chat_settings.json"
        runtime_log_path = logs_dir / f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        data_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return AppPaths(root, data_dir, logs_dir, config_path, runtime_log_path)

    def _set_safe_app_font(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        font = app.font()
        if font.pointSize() <= 0:
            font.setPointSize(11)
        else:
            font.setPointSize(font.pointSize() + 1)
        app.setFont(font)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        header = HeaderCard(
            "Groq Chat Desktop",
            "Σύγχρονη desktop εφαρμογή συνομιλίας με Groq API, χωρίς RAG και χωρίς Ollama. "
            "Επιλογή μοντέλου από Groq, ελληνικές απαντήσεις και αναλυτικά logs.",
        )
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        # ---------- Αριστερό panel ----------
        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_panel.setFrameShape(QFrame.NoFrame)
        left_content = QWidget()
        left_panel.setWidget(left_content)
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(12)

        connection_box = QGroupBox("Σύνδεση Groq")
        connection_layout = QVBoxLayout(connection_box)
        connection_layout.setContentsMargins(14, 24, 14, 14)
        connection_layout.setSpacing(10)

        api_form = QFormLayout()
        api_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        api_form.setFormAlignment(Qt.AlignTop)
        api_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        api_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        api_form.setHorizontalSpacing(12)
        api_form.setVerticalSpacing(10)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Επικόλλησε εδώ το GROQ_API_KEY")
        self.api_key_edit.setClearButtonEnabled(True)
        api_form.addRow("API Key:", self.api_key_edit)
        connection_layout.addLayout(api_form)

        key_row = QHBoxLayout()
        self.show_key_btn = QPushButton("Εμφάνιση")
        self.show_key_btn.setObjectName("SecondaryButton")
        self.save_settings_btn = QPushButton("Αποθήκευση ρυθμίσεων")
        self.save_settings_btn.setObjectName("SuccessButton")
        self.refresh_models_btn = QPushButton("Ανανέωση μοντέλων")
        self.refresh_models_btn.setObjectName("AccentButton")
        key_row.addWidget(self.show_key_btn)
        key_row.addWidget(self.save_settings_btn, 1)
        key_row.addWidget(self.refresh_models_btn, 1)
        connection_layout.addLayout(key_row)

        theme_row = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.toggle_theme_btn = QPushButton("Εναλλαγή theme")
        self.toggle_theme_btn.setObjectName("SecondaryButton")
        theme_row.addWidget(theme_label)
        theme_row.addWidget(self.theme_combo, 1)
        theme_row.addWidget(self.toggle_theme_btn)
        connection_layout.addLayout(theme_row)

        self.connection_status = QLabel("Χωρίς έλεγχο σύνδεσης ακόμη.")
        self.connection_status.setObjectName("StatusPill")
        self.connection_status.setObjectName("InfoLabel")
        self.connection_status.setWordWrap(True)
        connection_layout.addWidget(self.connection_status)
        left_layout.addWidget(connection_box)

        models_box = QGroupBox("Μοντέλα")
        models_layout = QVBoxLayout(models_box)
        models_layout.setContentsMargins(14, 24, 14, 14)
        models_layout.setSpacing(10)

        self.chat_only_checkbox = QCheckBox("Εμφάνιση κυρίως chat μοντέλων")
        self.chat_only_checkbox.setChecked(True)
        self.model_search_edit = QLineEdit()
        self.model_search_edit.setPlaceholderText("Αναζήτηση μοντέλου...")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(42)
        self.model_combo.setMinimumContentsLength(28)
        self.model_info_label = QLabel("Δεν έχουν φορτωθεί ακόμη μοντέλα.")
        self.model_info_label.setObjectName("ModelInfoCard")
        self.model_info_label.setObjectName("InfoLabel")
        self.model_info_label.setWordWrap(True)
        models_layout.addWidget(self.chat_only_checkbox)
        models_layout.addWidget(self.model_search_edit)
        models_layout.addWidget(self.model_combo)
        models_layout.addWidget(self.model_info_label)
        left_layout.addWidget(models_box)

        params_box = QGroupBox("Παράμετροι δημιουργίας")
        params_layout = QFormLayout(params_box)
        params_layout.setContentsMargins(14, 24, 14, 14)
        params_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        params_layout.setFormAlignment(Qt.AlignTop)
        params_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        params_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        params_layout.setHorizontalSpacing(12)
        params_layout.setVerticalSpacing(10)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 131072)
        self.max_tokens_spin.setValue(8192)
        self.max_tokens_spin.setSingleStep(256)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(0.40)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.95)

        self.greek_checkbox = QCheckBox("Προτίμηση για ελληνικές απαντήσεις")
        self.greek_checkbox.setChecked(True)
        self.greek_checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        params_layout.addRow("MaxTokens:", self.max_tokens_spin)
        params_layout.addRow("Temperature:", self.temperature_spin)
        params_layout.addRow("Top‑p:", self.top_p_spin)
        params_layout.addRow("Γλώσσα:", self.greek_checkbox)
        left_layout.addWidget(params_box)

        prompt_box = QGroupBox("System Prompt")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.setContentsMargins(14, 24, 14, 14)
        prompt_layout.setSpacing(10)
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setObjectName("SystemPromptEdit")
        self.system_prompt_edit.setMinimumHeight(130)
        self.system_prompt_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.system_prompt_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.system_prompt_edit.setPlaceholderText("Προαιρετικό system prompt...")
        self.system_prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT_EL)
        prompt_layout.addWidget(self.system_prompt_edit)
        left_layout.addWidget(prompt_box)

        logs_box = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_box)
        logs_layout.setContentsMargins(14, 24, 14, 14)
        logs_layout.setSpacing(10)
        self.logs_edit = QPlainTextEdit()
        self.logs_edit.setObjectName("LogsEdit")
        self.logs_edit.setReadOnly(True)
        self.logs_edit.setMinimumHeight(210)
        self.logs_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.logs_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.logs_edit.setMaximumBlockCount(2000)
        logs_layout.addWidget(self.logs_edit)
        left_layout.addWidget(logs_box, 1)

        splitter.addWidget(left_panel)

        # ---------- Δεξί panel ----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(12)

        conversation_box = QGroupBox("Απάντηση και ιστορικό συνομιλίας")
        conversation_box.setObjectName("ConversationBox")
        conversation_layout = QVBoxLayout(conversation_box)
        conversation_layout.setContentsMargins(8, 20, 8, 8)
        conversation_layout.setSpacing(6)
        self.chat_transcript = QTextBrowser()
        self.chat_transcript.setObjectName("ConversationView")
        self.chat_transcript.setOpenExternalLinks(False)
        self.chat_transcript.setReadOnly(True)
        self.chat_transcript.setMinimumHeight(1372)
        self.chat_transcript.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_transcript.setLineWrapMode(QTextEdit.WidgetWidth)
        self.chat_transcript.setPlaceholderText("Η συνομιλία και οι απαντήσεις του μοντέλου θα εμφανίζονται εδώ.")
        self.chat_transcript.setFrameShape(QFrame.NoFrame)
        self.chat_transcript.document().setDocumentMargin(2)
        conversation_layout.addWidget(self.chat_transcript)
        right_layout.addWidget(conversation_box, 6)
        self.answer_view = self.chat_transcript

        answer_actions_frame = QFrame()
        answer_actions_frame.setObjectName("AnswerActionsFrame")
        answer_actions_outer = QVBoxLayout(answer_actions_frame)
        answer_actions_outer.setContentsMargins(12, 10, 12, 10)
        answer_actions_outer.setSpacing(10)

        answer_actions = QHBoxLayout()
        answer_actions.setContentsMargins(0, 0, 0, 0)
        answer_actions.setSpacing(10)
        self.copy_answer_btn = QPushButton("Αντιγραφή απάντησης")
        self.copy_answer_btn.setObjectName("SecondaryButton")
        self.export_chat_btn = QPushButton("Εξαγωγή συνομιλίας")
        self.export_chat_btn.setObjectName("SuccessButton")
        self.clear_chat_btn = QPushButton("Καθαρισμός")
        self.clear_chat_btn.setObjectName("DangerButton")
        answer_actions.addWidget(self.copy_answer_btn)
        answer_actions.addWidget(self.export_chat_btn)
        answer_actions.addStretch(1)
        answer_actions.addWidget(self.clear_chat_btn)
        answer_actions_outer.addLayout(answer_actions)

        answer_code_row = QHBoxLayout()
        answer_code_row.setContentsMargins(0, 0, 0, 0)
        answer_code_row.setSpacing(10)
        self.code_blocks_label = QLabel("Blocks κώδικα:")
        self.code_blocks_label.setObjectName("InfoLabel")
        self.code_blocks_combo = QComboBox()
        self.code_blocks_combo.setMinimumHeight(40)
        self.code_blocks_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.copy_code_btn = QPushButton("Αντιγραφή επιλεγμένου κώδικα")
        self.copy_code_btn.setObjectName("AccentButton")
        self.copy_code_btn.setEnabled(False)
        answer_code_row.addWidget(self.code_blocks_label)
        answer_code_row.addWidget(self.code_blocks_combo, 1)
        answer_code_row.addWidget(self.copy_code_btn)
        answer_actions_outer.addLayout(answer_code_row)

        right_layout.addWidget(answer_actions_frame)

        ask_box = QGroupBox("Ερώτηση")
        ask_box.setObjectName("QuestionBox")
        ask_layout = QVBoxLayout(ask_box)
        ask_layout.setContentsMargins(8, 20, 8, 8)
        ask_layout.setSpacing(6)
        self.question_edit = QTextEdit()
        self.question_edit.setObjectName("QuestionEdit")
        self.question_edit.setPlaceholderText("Γράψε την ερώτησή σου εδώ...")
        self.question_edit.setMinimumHeight(190)
        self.question_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.question_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.question_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.question_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.question_edit.setFrameShape(QFrame.NoFrame)
        self.question_edit.document().setDocumentMargin(2)
        ask_layout.addWidget(self.question_edit)
        right_layout.addWidget(ask_box, 2)

        ask_actions_frame = QFrame()
        ask_actions_frame.setObjectName("QuestionActionsFrame")
        ask_actions = QHBoxLayout(ask_actions_frame)
        ask_actions.setContentsMargins(12, 10, 12, 10)
        ask_actions.setSpacing(10)
        self.send_btn = QPushButton("Αποστολή")
        self.send_btn.setObjectName("PrimaryButton")
        self.stop_btn = QPushButton("Διακοπή")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        ask_actions.addStretch(1)
        ask_actions.addWidget(self.stop_btn)
        ask_actions.addWidget(self.send_btn)
        right_layout.addWidget(ask_actions_frame)

        progress_frame = QFrame()
        progress_frame.setObjectName("ProgressFrame")
        progress_layout = QHBoxLayout(progress_frame)
        progress_layout.setContentsMargins(12, 8, 12, 8)
        progress_layout.setSpacing(10)
        self.progress_label = QLabel("Έτοιμο.")
        self.progress_label.setObjectName("ProgressLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar, 1)
        right_layout.addWidget(progress_frame)

        splitter.addWidget(right_panel)
        splitter.setSizes([420, 980])

        status = QStatusBar()
        self.setStatusBar(status)

        self._update_transcript_view()
        self._refresh_code_blocks_ui()

    def _apply_theme(self) -> None:
        light_qss = """
            QWidget {
                background: #f3f7fc;
                color: #17263c;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
                background: #f3f7fc;
            }
            QToolTip {
                background: #0f172a;
                color: #ffffff;
                border: 1px solid #1d4ed8;
                padding: 6px 8px;
            }
            QTextEdit, QTextBrowser, QPlainTextEdit {
                background: #fdfefe;
                border: 1px solid #cfdced;
                border-radius: 14px;
                padding: 10px 12px;
                selection-background-color: #2563eb;
            }
            QTextEdit#QuestionEdit, QTextBrowser#ConversationView, QTextBrowser#AnswerView, QTextBrowser#ChatTranscript {
                background: transparent;
                border: none;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
            }
            QGroupBox#QuestionBox, QGroupBox#ConversationBox {
                padding: 8px 6px 6px 6px;
            }
            QScrollBar:vertical {
                background: #e8eef8;
                width: 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #8da7cf;
                min-height: 26px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6c8fc6;
            }
            QScrollBar:horizontal {
                background: #e8eef8;
                height: 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #8da7cf;
                min-width: 26px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #6c8fc6;
            }
            QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
                background: transparent;
                border: none;
            }
            #HeaderCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:0.55 #eff5ff, stop:1 #e6f0ff);
                border: 1px solid #d5e1f2;
                border-radius: 20px;
            }
            #HeroTitle {
                font-size: 30px;
                font-weight: 800;
                color: #0f2240;
                letter-spacing: 0.2px;
            }
            #HeroSubtitle {
                font-size: 15px;
                color: #5c6f8b;
                line-height: 1.45em;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7e3f2;
                border-radius: 16px;
                margin-top: 14px;
                padding: 14px 12px 12px 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 2px 8px;
                color: #28456e;
                background: #eef4ff;
                border: 1px solid #d3e0f5;
                border-radius: 8px;
            }
            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox {
                background: #fbfdff;
                border: 1px solid #ccdaee;
                border-radius: 12px;
                padding: 11px 13px;
                color: #16273c;
                font-size: 15px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #5b8cff;
                background: #ffffff;
            }
            QComboBox::drop-down, QSpinBox::down-button, QSpinBox::up-button, QDoubleSpinBox::down-button, QDoubleSpinBox::up-button {
                width: 28px;
                border: none;
                background: transparent;
            }
            QPushButton {
                background: #edf3ff;
                color: #173052;
                border: 1px solid #c6d7f6;
                border-radius: 12px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #e4edff;
                border-color: #a8c2f0;
            }
            QPushButton:pressed {
                background: #dae7ff;
            }
            QPushButton:disabled {
                background: #edf1f7;
                color: #7c8fa9;
                border-color: #d4ddea;
            }
            QPushButton#PrimaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #3b82f6);
                color: #ffffff;
                border: 1px solid #2f67df;
            }
            QPushButton#PrimaryButton:hover { background: #2d6ff1; }
            QPushButton#PrimaryButton:pressed { background: #1f57d2; }
            QPushButton#SecondaryButton {
                background: #eef4ff;
                color: #21456e;
                border: 1px solid #c9daf6;
            }
            QPushButton#AccentButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7aa2ff, stop:1 #5b8cff);
                color: #ffffff;
                border: 1px solid #5d8fec;
            }
            QPushButton#AccentButton:hover { background: #6794ff; }
            QPushButton#SuccessButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5a7fb9, stop:1 #7296d1);
                color: #ffffff;
                border: 1px solid #6487be;
            }
            QPushButton#SuccessButton:hover { background: #6f93cb; }
            QPushButton#DangerButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6b7280, stop:1 #94a3b8);
                color: #ffffff;
                border: 1px solid #7a8597;
            }
            QPushButton#DangerButton:hover { background: #7b8595; }
            QCheckBox {
                spacing: 8px;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #a9bfdc;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #2563eb;
                border: 1px solid #2563eb;
            }
            QLabel {
                font-size: 14px;
            }
            QLabel#InfoLabel {
                color: #647891;
            }
            QLabel#StatusPill, QLabel#ModelInfoCard {
                background: #f6faff;
                border: 1px solid #d7e4f5;
                border-radius: 10px;
                padding: 10px 12px;
                color: #445d7c;
            }
            QLabel#ProgressLabel {
                color: #29405f;
                font-weight: 700;
            }
            #ProgressFrame, #AnswerActionsFrame {
                background: #ffffff;
                border: 1px solid #d7e3f2;
                border-radius: 14px;
            }
            QProgressBar {
                background: #edf4fd;
                border: 1px solid #d0ddef;
                border-radius: 9px;
                min-height: 16px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f8cff, stop:1 #7aa2ff);
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 4px 0 4px 0;
            }
            QScrollBar::handle:vertical {
                background: #c5d5ea;
                min-height: 28px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a9bfdc;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 12px;
                margin: 0 4px 0 4px;
            }
            QScrollBar::handle:horizontal {
                background: #c5d5ea;
                min-width: 28px;
                border-radius: 6px;
            }
            QSplitter::handle {
                background: #dce6f4;
                width: 8px;
                margin: 4px 0;
                border-radius: 4px;
            }
            QStatusBar {
                background: #ecf3fb;
                color: #445975;
                border-top: 1px solid #d6e2f1;
            }
            QTextBrowser, QPlainTextEdit, QTextEdit {
                padding-top: 12px;
                padding-bottom: 12px;
            }
        """
        dark_qss = """
            QWidget {
                background: #0b1220;
                color: #e5eefc;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
                background: #0b1220;
            }
            QToolTip {
                background: #111827;
                color: #ffffff;
                border: 1px solid #2563eb;
                padding: 6px 8px;
            }
            QTextEdit, QTextBrowser, QPlainTextEdit {
                background: #111a2b;
                border: 1px solid #233450;
                border-radius: 14px;
                padding: 10px 12px;
                selection-background-color: #2563eb;
            }
            QTextEdit#QuestionEdit, QTextBrowser#ConversationView, QTextBrowser#AnswerView, QTextBrowser#ChatTranscript {
                background: transparent;
                border: none;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
            }
            QGroupBox#QuestionBox, QGroupBox#ConversationBox {
                padding: 8px 6px 6px 6px;
            }
            QScrollBar:vertical {
                background: #121d30;
                width: 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #4a6797;
                min-height: 26px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6281b2;
            }
            QScrollBar:horizontal {
                background: #121d30;
                height: 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #4a6797;
                min-width: 26px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #6281b2;
            }
            QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
                background: transparent;
                border: none;
            }
            #HeaderCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111a2c, stop:0.55 #13213a, stop:1 #10314a);
                border: 1px solid #223252;
                border-radius: 20px;
            }
            #HeroTitle {
                font-size: 30px;
                font-weight: 800;
                color: #f8fbff;
            }
            #HeroSubtitle {
                font-size: 15px;
                color: #bfd0e7;
                line-height: 1.45em;
            }
            QGroupBox {
                background: #0f1728;
                border: 1px solid #223252;
                border-radius: 16px;
                margin-top: 14px;
                padding: 14px 12px 12px 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 2px 8px;
                color: #d8e4f7;
                background: #13243d;
                border: 1px solid #284267;
                border-radius: 8px;
            }
            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox {
                background: #0b1322;
                border: 1px solid #2a3a5d;
                border-radius: 12px;
                padding: 11px 13px;
                color: #f3f7ff;
                font-size: 15px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #5b8cff;
                background: #0d1628;
            }
            QComboBox::drop-down, QSpinBox::down-button, QSpinBox::up-button, QDoubleSpinBox::down-button, QDoubleSpinBox::up-button {
                width: 28px;
                border: none;
                background: transparent;
            }
            QPushButton {
                background: #18253c;
                color: #e8f0ff;
                border: 1px solid #294061;
                border-radius: 12px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #1d2d49;
                border-color: #36547e;
            }
            QPushButton:pressed {
                background: #13243d;
            }
            QPushButton:disabled {
                background: #22314d;
                color: #8da1c0;
                border-color: #2a3a5d;
            }
            QPushButton#PrimaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #3b82f6);
                color: white;
                border: 1px solid #3b82f6;
            }
            QPushButton#PrimaryButton:hover { background: #2d6df0; }
            QPushButton#SecondaryButton {
                background: #162338;
                color: #dce9fb;
                border: 1px solid #304768;
            }
            QPushButton#AccentButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d6fd8, stop:1 #5b8cff);
                color: white;
                border: 1px solid #4f7fe0;
            }
            QPushButton#AccentButton:hover { background: #537fe6; }
            QPushButton#SuccessButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5c77b5, stop:1 #7296d1);
                color: white;
                border: 1px solid #6487be;
            }
            QPushButton#SuccessButton:hover { background: #6f93cb; }
            QPushButton#DangerButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5b6474, stop:1 #94a3b8);
                color: white;
                border: 1px solid #6d788b;
            }
            QPushButton#DangerButton:hover { background: #7a879d; }
            QCheckBox {
                spacing: 8px;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #456089;
                background: #0f1728;
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
                border: 1px solid #3b82f6;
            }
            QLabel {
                font-size: 14px;
            }
            QLabel#InfoLabel {
                color: #9eb0cc;
            }
            QLabel#StatusPill, QLabel#ModelInfoCard {
                background: #111d30;
                border: 1px solid #243552;
                border-radius: 10px;
                padding: 10px 12px;
                color: #b7c9e6;
            }
            QLabel#ProgressLabel {
                color: #d8e4f7;
                font-weight: 700;
            }
            #ProgressFrame, #AnswerActionsFrame {
                background: #0f1728;
                border: 1px solid #223252;
                border-radius: 14px;
            }
            QProgressBar {
                background: #0b1322;
                border: 1px solid #2a3a5d;
                border-radius: 9px;
                min-height: 16px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f8cff, stop:1 #7aa2ff);
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 4px 0 4px 0;
            }
            QScrollBar::handle:vertical {
                background: #314562;
                min-height: 28px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #446082;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 12px;
                margin: 0 4px 0 4px;
            }
            QScrollBar::handle:horizontal {
                background: #314562;
                min-width: 28px;
                border-radius: 6px;
            }
            QSplitter::handle {
                background: #223252;
                width: 8px;
                margin: 4px 0;
                border-radius: 4px;
            }
            QStatusBar {
                background: #0d1525;
                color: #bed1ee;
                border-top: 1px solid #223252;
            }
            QTextBrowser, QPlainTextEdit, QTextEdit {
                padding-top: 12px;
                padding-bottom: 12px;
            }
        """
        self.setStyleSheet(light_qss if self.current_theme == "light" else dark_qss)

    def _connect_signals(self) -> None:
        self.show_key_btn.clicked.connect(self._toggle_api_key_visibility)
        self.save_settings_btn.clicked.connect(self._save_settings)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.toggle_theme_btn.clicked.connect(self._toggle_theme)
        self.refresh_models_btn.clicked.connect(lambda: self.refresh_models(auto=False))
        self.model_search_edit.textChanged.connect(self._update_model_filter)
        self.chat_only_checkbox.toggled.connect(self._update_model_filter)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.send_btn.clicked.connect(self.send_message)
        self.stop_btn.clicked.connect(self.stop_current_task)
        self.copy_answer_btn.clicked.connect(self.copy_last_answer)
        self.copy_code_btn.clicked.connect(self.copy_selected_code_block)
        self.export_chat_btn.clicked.connect(self.export_chat)
        self.clear_chat_btn.clicked.connect(self.clear_chat)

    def _load_settings(self) -> None:
        env_key = os.environ.get("GROQ_API_KEY", "")
        settings = {}
        if self.paths.config_path.exists():
            try:
                settings = json.loads(self.paths.config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._log_error("Αποτυχία φόρτωσης ρυθμίσεων.", {"traceback": traceback.format_exc()})

        self.api_key_edit.setText(settings.get("api_key") or env_key)
        theme = str(settings.get("theme", "light")).lower()
        if theme not in {"light", "dark"}:
            theme = "light"
        self.current_theme = theme
        idx = self.theme_combo.findData(theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.model_search_edit.setText(settings.get("model_search", ""))
        self.chat_only_checkbox.setChecked(bool(settings.get("chat_only", True)))
        self.temperature_spin.setValue(float(settings.get("temperature", 0.40)))
        self.top_p_spin.setValue(float(settings.get("top_p", 0.95)))
        self.max_tokens_spin.setValue(int(settings.get("max_tokens", 8192)))
        self.greek_checkbox.setChecked(bool(settings.get("prefer_greek", True)))
        self.system_prompt_edit.setPlainText(settings.get("system_prompt") or DEFAULT_SYSTEM_PROMPT_EL)
        self._wanted_model_after_refresh = settings.get("selected_model", "")

    def _save_settings(self) -> None:
        settings = {
            "api_key": self.api_key_edit.text().strip(),
            "selected_model": self.model_combo.currentData() or self.model_combo.currentText(),
            "model_search": self.model_search_edit.text().strip(),
            "chat_only": self.chat_only_checkbox.isChecked(),
            "temperature": self.temperature_spin.value(),
            "top_p": self.top_p_spin.value(),
            "max_tokens": self.max_tokens_spin.value(),
            "prefer_greek": self.greek_checkbox.isChecked(),
            "system_prompt": self.system_prompt_edit.toPlainText().strip(),
            "theme": self.current_theme,
        }
        try:
            self.paths.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.paths.config_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log_info("Οι ρυθμίσεις αποθηκεύτηκαν.")
            self.statusBar().showMessage("Οι ρυθμίσεις αποθηκεύτηκαν.", 5000)
        except Exception:
            self._log_error("Αποτυχία αποθήκευσης ρυθμίσεων.", {"traceback": traceback.format_exc()})
            QMessageBox.critical(self, "Σφάλμα", "Δεν ήταν δυνατή η αποθήκευση των ρυθμίσεων.")

    def _toggle_api_key_visibility(self) -> None:
        if self.api_key_edit.echoMode() == QLineEdit.Password:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("Απόκρυψη")
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("Εμφάνιση")

    def _on_theme_changed(self, index: int) -> None:
        theme = self.theme_combo.itemData(index) if index >= 0 else "light"
        theme = theme or "light"
        if theme == self.current_theme:
            return
        self.current_theme = str(theme)
        self._apply_theme()
        self._update_transcript_view()
        self.statusBar().showMessage(f"Ενεργό theme: {'Light' if self.current_theme == 'light' else 'Dark'}", 4000)

    def _toggle_theme(self) -> None:
        next_theme = "dark" if self.current_theme == "light" else "light"
        idx = self.theme_combo.findData(next_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)

    def refresh_models(self, *, auto: bool) -> None:
        if self.model_refresh_thread and self.model_refresh_thread.isRunning():
            return

        api_key = self.api_key_edit.text().strip()
        if not api_key:
            if not auto:
                QMessageBox.warning(self, "Groq API Key", "Συμπλήρωσε πρώτα το Groq API key.")
            self.connection_status.setText("Δεν υπάρχει API key.")
            return

        self._set_busy(True, "models", "Ανανέωση λίστας μοντέλων...")
        self.connection_status.setText(f"Χρήση API key: {mask_api_key(api_key)}")
        self.model_refresh_thread = ModelRefreshThread(api_key)
        self.model_refresh_thread.log.connect(self._log_info)
        self.model_refresh_thread.success.connect(self._on_models_loaded)
        self.model_refresh_thread.error.connect(self._on_models_error)
        self.model_refresh_thread.finished.connect(lambda: self._set_busy(False, "models", "Έτοιμο."))
        self.model_refresh_thread.start()

    def _on_models_loaded(self, models: List[Dict[str, Any]]) -> None:
        self.models_raw = sorted(models, key=lambda m: str(m.get("id", "")).lower())
        self._update_model_filter()
        total = len(self.models_raw)
        chat_like = sum(1 for m in self.models_raw if guess_chat_model(str(m.get("id", ""))))
        self.connection_status.setText(
            f"Επιτυχής ανάγνωση Groq models. Σύνολο: {total} • Εκτιμώμενα chat μοντέλα: {chat_like}"
        )
        self._log_info("Η λίστα μοντέλων ενημερώθηκε.", {"total_models": total, "chat_like_models": chat_like})
        self.statusBar().showMessage("Η λίστα μοντέλων ενημερώθηκε.", 5000)

    def _on_models_error(self, message: str, details: Dict[str, Any]) -> None:
        self._log_error(message, details)
        self.connection_status.setText("Αποτυχία ανάκτησης μοντέλων. Δες τα logs.")
        QMessageBox.critical(self, "Σφάλμα ανάκτησης μοντέλων", f"{message}\n\nΔες τα logs για λεπτομέρειες.")

    def _update_model_filter(self) -> None:
        current = self.model_combo.currentData() or self.model_combo.currentText()
        search = self.model_search_edit.text().strip().lower()
        chat_only = self.chat_only_checkbox.isChecked()

        filtered: List[Dict[str, Any]] = []
        for model in self.models_raw:
            model_id = str(model.get("id", "")).strip()
            if not model_id:
                continue
            if chat_only and not guess_chat_model(model_id):
                continue
            if search and search not in model_id.lower():
                continue
            filtered.append(model)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in filtered:
            model_id = str(model.get("id", "")).strip()
            self.model_combo.addItem(model_id, model_id)
        self.model_combo.blockSignals(False)

        wanted = getattr(self, "_wanted_model_after_refresh", "") or current
        if wanted:
            idx = self.model_combo.findData(wanted)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self._wanted_model_after_refresh = ""
            elif self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)

        if self.model_combo.count() == 0:
            self.model_info_label.setText("Δεν βρέθηκαν μοντέλα με το τρέχον φίλτρο.")
        else:
            self._on_model_changed(self.model_combo.currentIndex())

    def _on_model_changed(self, index: int) -> None:
        if index < 0:
            self.model_info_label.setText("Δεν έχει επιλεγεί μοντέλο.")
            return
        model_id = self.model_combo.itemData(index) or self.model_combo.itemText(index)
        model = next((m for m in self.models_raw if str(m.get("id")) == str(model_id)), None)
        if not model:
            self.model_info_label.setText(str(model_id))
            return
        details = {
            "id": model.get("id"),
            "owned_by": model.get("owned_by"),
            "created": model.get("created"),
            "object": model.get("object"),
        }
        details_text = " • ".join(f"{k}: {v}" for k, v in details.items() if v is not None)
        self.model_info_label.setText(details_text or str(model_id))

    def _build_messages_for_request(self, user_text: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        if self.greek_checkbox.isChecked() and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def send_message(self) -> None:
        if self.chat_thread and self.chat_thread.isRunning():
            return

        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Groq API Key", "Συμπλήρωσε πρώτα το Groq API key.")
            return

        model = self.model_combo.currentData() or self.model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(self, "Μοντέλο", "Επίλεξε πρώτα μοντέλο από τη λίστα.")
            return

        user_text = self.question_edit.toPlainText().strip()
        if not user_text:
            QMessageBox.information(self, "Ερώτηση", "Γράψε πρώτα μια ερώτηση.")
            return

        messages = self._build_messages_for_request(user_text)
        self._append_chat_item("user", user_text)
        self.question_edit.clear()
        self.pending_assistant_text = "Περιμένω απάντηση από το Groq..."
        self._update_transcript_view()

        self._set_busy(True, "chat", "Αποστολή ερώτησης στο Groq...")
        self.chat_thread = ChatThread(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=self.temperature_spin.value(),
            top_p=self.top_p_spin.value(),
            max_completion_tokens=self.max_tokens_spin.value(),
        )
        self.chat_thread.log.connect(self._log_info)
        self.chat_thread.success.connect(self._on_chat_success)
        self.chat_thread.error.connect(self._on_chat_error)
        self.chat_thread.finished.connect(lambda: self._set_busy(False, "chat", "Έτοιμο."))
        self.chat_thread.start()

    def _on_chat_success(self, result: Dict[str, Any]) -> None:
        text = result.get("text", "").strip()
        raw = result.get("raw")
        request_info = result.get("request")
        self.last_answer_text = text
        self.pending_assistant_text = ""
        self.last_code_blocks = _extract_code_blocks(text)
        self._append_chat_item("assistant", text)
        self._refresh_code_blocks_ui()
        self._log_info(
            "Επιτυχής απάντηση από Groq.",
            {
                "request": request_info,
                "response_usage": raw.get("usage") if isinstance(raw, dict) else None,
                "response_model": raw.get("model") if isinstance(raw, dict) else None,
            },
        )
        self.statusBar().showMessage("Η απάντηση λήφθηκε επιτυχώς.", 5000)

    def _on_chat_error(self, message: str, details: Dict[str, Any]) -> None:
        self._log_error(message, details)
        self.last_answer_text = ""
        self.pending_assistant_text = visible_message
        self.last_code_blocks = []
        visible_message = (
            f"Σφάλμα Groq: {message}\n\n"
            "Άνοιξε τα logs για ακριβείς λεπτομέρειες (endpoint, status code, body, traceback)."
        )
        self._update_transcript_view()
        self._refresh_code_blocks_ui()
        QMessageBox.critical(self, "Σφάλμα απάντησης", visible_message)

    def stop_current_task(self) -> None:
        stopped_any = False
        if self.model_refresh_thread and self.model_refresh_thread.isRunning():
            self.model_refresh_thread.requestInterruption()
            self.model_refresh_thread.terminate()
            self.model_refresh_thread.wait(1500)
            stopped_any = True
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.requestInterruption()
            self.chat_thread.terminate()
            self.chat_thread.wait(1500)
            stopped_any = True

        self._set_busy(False, self.current_busy_scope, "Η εργασία διακόπηκε.")
        if stopped_any:
            self.pending_assistant_text = "Η εργασία διακόπηκε από τον χρήστη."
            self._update_transcript_view()
            self._log_info("Η τρέχουσα εργασία διακόπηκε από τον χρήστη.")
            self.statusBar().showMessage("Η εργασία διακόπηκε.", 5000)

    def clear_chat(self) -> None:
        self.chat_history.clear()
        self.last_answer_text = ""
        self.pending_assistant_text = ""
        self.last_code_blocks = []
        self._update_transcript_view()
        self._refresh_code_blocks_ui()
        self.statusBar().showMessage("Το ιστορικό καθαρίστηκε.", 4000)

    def copy_last_answer(self) -> None:
        text = self.last_answer_text.strip()
        if not text:
            QMessageBox.information(self, "Αντιγραφή", "Δεν υπάρχει απάντηση για αντιγραφή.")
            return
        QGuiApplication.clipboard().setText(text)
        self.statusBar().showMessage("Η απάντηση αντιγράφηκε στο πρόχειρο.", 4000)

    def copy_selected_code_block(self) -> None:
        index = self.code_blocks_combo.currentIndex()
        if index < 0 or index >= len(self.last_code_blocks):
            QMessageBox.information(self, "Αντιγραφή κώδικα", "Δεν υπάρχει επιλεγμένο code block για αντιγραφή.")
            return
        lang, code = self.last_code_blocks[index]
        QGuiApplication.clipboard().setText(code)
        self.statusBar().showMessage(f"Ο κώδικας ({lang}) αντιγράφηκε στο πρόχειρο.", 4000)

    def _refresh_code_blocks_ui(self) -> None:
        self.code_blocks_combo.blockSignals(True)
        self.code_blocks_combo.clear()
        if not self.last_code_blocks:
            self.code_blocks_combo.addItem("Δεν υπάρχουν code blocks")
            self.code_blocks_combo.setEnabled(False)
            self.copy_code_btn.setEnabled(False)
            self.code_blocks_label.setText("Blocks κώδικα:")
        else:
            for idx, (lang, code) in enumerate(self.last_code_blocks, start=1):
                preview = code.strip().splitlines()[0] if code.strip() else "κενό block"
                preview = preview[:52] + ("…" if len(preview) > 52 else "")
                self.code_blocks_combo.addItem(f"{idx}. {lang} — {preview}")
            self.code_blocks_combo.setEnabled(True)
            self.copy_code_btn.setEnabled(True)
            self.code_blocks_label.setText(f"Blocks κώδικα: {len(self.last_code_blocks)}")
        self.code_blocks_combo.blockSignals(False)

    def export_chat(self) -> None:
        if not self.chat_history:
            QMessageBox.information(self, "Εξαγωγή", "Δεν υπάρχει συνομιλία για εξαγωγή.")
            return

        default_name = f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση συνομιλίας", str(self.paths.root / default_name), "Text files (*.txt)")
        if not path:
            return

        lines = [f"{APP_TITLE} v{APP_VERSION}", "=" * 60, ""]
        for item in self.chat_history:
            role = "Χρήστης" if item.get("role") == "user" else "Βοηθός"
            lines.append(f"[{role}]")
            lines.append(item.get("content", ""))
            lines.append("")

        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            self.statusBar().showMessage("Η συνομιλία αποθηκεύτηκε.", 5000)
        except Exception:
            self._log_error("Αποτυχία εξαγωγής συνομιλίας.", {"traceback": traceback.format_exc()})
            QMessageBox.critical(self, "Σφάλμα", "Δεν ήταν δυνατή η αποθήκευση του αρχείου.")

    def _append_chat_item(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})
        self._update_transcript_view()

    def _render_chat_bubble_html(self, role: str, content: str, *, pending: bool = False) -> str:
        palette = _theme_palette(self.current_theme)
        is_user = role == "user"
        bg = palette["user_bg"] if is_user else palette["assistant_bg"]
        border = palette["user_border"] if is_user else palette["assistant_border"]
        title_color = palette["user_title"] if is_user else palette["assistant_title"]
        title = "Χρήστης" if is_user else ("Βοηθός" if not pending else "Βοηθός • σε εξέλιξη")
        opacity = "0.90" if pending else "1.0"
        content_html = _render_rich_text_html(content or "", palette, font_px=14)
        return (
            f"<div style='margin:10px 0; padding:14px 16px; background:{bg}; border:1px solid {border}; border-radius:14px; opacity:{opacity};'>"
            f"<div style='font-weight:700; color:{title_color}; margin-bottom:8px;'>{title}</div>"
            f"<div style='color:{palette['text']}; line-height:1.6; white-space:normal;'>{content_html}</div>"
            f"</div>"
        )

    def _update_transcript_view(self) -> None:
        palette = _theme_palette(self.current_theme)
        has_history = bool(self.chat_history)
        has_pending = bool((self.pending_assistant_text or "").strip())
        if not has_history and not has_pending:
            self.chat_transcript.setHtml(
                f"<div style='color:{palette['hint']}; font-size:14px; padding:4px 2px;'>"
                "Η συνομιλία και οι απαντήσεις θα εμφανίζονται εδώ. Γράψε μια ερώτηση και πάτησε Αποστολή."
                "</div>"
            )
            return

        parts = ["<div style='font-family: Segoe UI, Arial, sans-serif; font-size:14px;'>"]
        for item in self.chat_history:
            parts.append(self._render_chat_bubble_html(item.get("role", "assistant"), item.get("content", "")))
        if has_pending:
            parts.append(self._render_chat_bubble_html("assistant", self.pending_assistant_text, pending=True))
        parts.append("</div>")
        self.chat_transcript.setHtml("".join(parts))
        self.chat_transcript.moveCursor(QTextCursor.End)

    def _update_answer_view(self, text: str) -> None:
        self.pending_assistant_text = text or ""
        self._update_transcript_view()

    def _set_busy(self, busy: bool, scope: Optional[str], message: str) -> None:
        self.current_busy_scope = scope if busy else None
        self.progress_label.setText(message)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

        # Περιορισμένο κλείδωμα: όχι όλο το UI.
        self.refresh_models_btn.setEnabled(not busy or scope != "models")
        self.model_combo.setEnabled(not busy or scope != "models")
        self.model_search_edit.setEnabled(not busy or scope != "models")
        self.chat_only_checkbox.setEnabled(not busy or scope != "models")

        self.send_btn.setEnabled(not busy or scope != "chat")
        self.question_edit.setEnabled(not busy or scope != "chat")
        self.stop_btn.setEnabled(busy)

    def _log_info(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = self.logger.write("info", message, details)
        self.logs_edit.appendPlainText(entry.rstrip())
        self.logs_edit.verticalScrollBar().setValue(self.logs_edit.verticalScrollBar().maximum())

    def _log_error(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = self.logger.write("error", message, details)
        self.logs_edit.appendPlainText(entry.rstrip())
        self.logs_edit.verticalScrollBar().setValue(self.logs_edit.verticalScrollBar().maximum())

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_settings()
        try:
            self.stop_current_task()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("OpenAI")
    app.setStyle("Fusion")

    font = app.font()
    if font.pointSize() <= 0:
        font.setPointSize(11)
    app.setFont(font)

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
