"""GUI wrapper for translate_gemini.py.

Features:
  1.  Drag & drop of XML files onto the window (if tkinterdnd2 is installed).
  2.  Real progress bar (determinate) with batch counter.
  3.  "Open output folder" button on completion.
  4.  Remembers last configuration (language, workers, last folder, window size).
  5.  Stop button to cancel mid-run.
  6.  Search/filter in the log console.
  7.  Pre-flight string counter and rough cost estimate.
  8.  Global dark theme.
  9.  API key validation button.
  10. Batch mode: queue multiple XML files to translate in sequence.
"""

from __future__ import annotations

import difflib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

# Try to enable drag & drop. Falls back gracefully if the package is missing.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# Import the translation module.
try:
    import translate_gemini as tg
except ImportError as exc:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing module / Falta el módulo",
        f"Could not import translate_gemini.py — keep it in the same folder as translate_gui.py.\n"
        f"No se pudo importar translate_gemini.py — debe estar en la misma carpeta que translate_gui.py.\n\n"
        f"Details / Detalles: {exc}",
    )
    sys.exit(1)


APP_TITLE = "Gemini XML Translator"
CONFIG_FILENAME = ".translate_gui_config.json"

# Rough cost per 1M tokens for gemini-2.5-flash (USD). Update if pricing changes.
# Output is ~8x input; translations are billed mostly on the OUTPUT side.
COST_PER_M_INPUT_TOKENS = 0.30
COST_PER_M_OUTPUT_TOKENS = 2.50
# Approximation for Latin scripts: each character ≈ 0.25 tokens.
CHARS_PER_TOKEN = 4
# Translations tend to run a bit longer than the source (+ JSON framing).
OUTPUT_LENGTH_FACTOR = 1.15


def _approx_tokens(text: str) -> float:
    """Rough token count: CJK chars ≈ 1 token each, the rest ≈ 4 chars/token."""
    cjk = 0
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF \
                or 0xAC00 <= cp <= 0xD7AF:
            cjk += 1
    return cjk + (len(text) - cjk) / CHARS_PER_TOKEN

DEFAULT_TARGET_OPTIONS = [
    "Latin American Spanish",
    "Spanish (Spain)",
    "French",
    "German",
    "Italian",
    "Portuguese (Brazil)",
    "Portuguese (Portugal)",
    "Japanese",
    "Korean",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Russian",
    "Arabic",
    "Turkish",
]

DEFAULT_SOURCE_OPTIONS = ["English"] + DEFAULT_TARGET_OPTIONS

# Short codes used to partition the automatic output/cache filenames per language
# pair (e.g. "_zh-en"). Checked in order, so variant names must come before their
# base language. Fallback for unknown names: first word, letters only.
LANG_SLUGS = [
    ("spain", "es-es"), ("spanish", "es"), ("español", "es"), ("espanol", "es"),
    ("traditional", "zh-tw"), ("chinese", "zh"),
    ("brazil", "pt-br"), ("portugal", "pt-pt"), ("portuguese", "pt"),
    ("english", "en"), ("french", "fr"), ("german", "de"), ("italian", "it"),
    ("japanese", "ja"), ("korean", "ko"), ("russian", "ru"),
    ("arabic", "ar"), ("turkish", "tr"),
]


def _lang_slug(name: str) -> str:
    lowered = (name or "").strip().lower()
    for key, code in LANG_SLUGS:
        if key in lowered:
            return code
    first_word = re.sub(r"[^a-z]", "", lowered.split()[0]) if lowered.split() else ""
    return first_word or "xx"


# Slug → dropdown display name, for the source-language-mismatch dialog.
SLUG_TO_NAME = {
    "en": "English", "es": "Latin American Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese (Brazil)", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese (Simplified)", "ru": "Russian", "ar": "Arabic", "tr": "Turkish",
}

# Function-word votes to tell Latin-script languages apart in _detect_language.
_STOPWORD_VOTES = {
    "en": {"the", "and", "of", "to", "is", "you", "your", "with", "for", "this"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "ein", "eine", "für", "wird"},
    "es": {"el", "la", "los", "las", "de", "que", "para", "con", "una", "tu", "más"},
    "fr": {"le", "les", "des", "et", "est", "pour", "une", "vous", "dans", "plus"},
    "it": {"il", "che", "di", "non", "per", "una", "gli", "sono", "più", "è",
           "della", "delle", "degli", "questa", "questo"},
    "pt": {"o", "os", "de", "que", "não", "para", "uma", "você", "com", "mais"},
    "tr": {"ve", "bir", "için", "bu", "daha", "olarak"},
}


def _detect_language(texts) -> Optional[str]:
    """Best-effort guess of the dominant language of `texts` (a slug like 'en'), or None.

    Conservative by design: character scripts decide the non-Latin languages;
    Latin-script ones need a clear stopword-vote margin, otherwise None (stay quiet).
    """
    sample = [t for t in texts if t and t.strip()][:200]
    if not sample:
        return None
    joined = " ".join(sample)

    kana = han = hangul = cyrillic = arabic = latin = 0
    for ch in joined:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            kana += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            han += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            hangul += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin += 1

    # Kana implies Japanese even mixed with han; han alone means Chinese.
    if kana > 5:
        return "ja"
    for slug, count in (("ko", hangul), ("zh", han), ("ru", cyrillic), ("ar", arabic)):
        # Markup/placeholders add Latin chars even in non-Latin files, hence the soft ratio.
        if count > 5 and count * 2 >= latin:
            return slug

    words = re.findall(r"[a-záéíóúüñàèìòùâêîôûäöüßçãõ]+", joined.lower())
    if len(words) < 20:
        return None
    votes = {slug: 0 for slug in _STOPWORD_VOTES}
    for word in words:
        for slug, stopwords in _STOPWORD_VOTES.items():
            if word in stopwords:
                votes[slug] += 1
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    (best_slug, best), (_, second) = ranked[0], ranked[1]
    # Romance languages share function words ("la", "de", "una"), so demand a
    # clear but not extreme lead; anything murkier returns None and we stay quiet.
    if best >= max(8, len(words) // 50) and best >= second * 1.5:
        return best_slug
    return None

# Dark theme palette.
BG_MAIN = "#1e1e1e"
BG_FRAME = "#252525"
BG_ENTRY = "#2d2d2d"
BG_LOG = "#1a1a1a"
FG_MAIN = "#d4d4d4"
FG_MUTED = "#888888"
FG_ACCENT = "#4ec9b0"
FG_WARN = "#dcdcaa"
FG_ERROR = "#f48771"
FG_SUCCESS = "#6a9955"

# Available UI languages: (display name, code).
UI_LANGUAGES = [("Español", "es"), ("English", "en")]

# All user-facing UI text. self.t(key, **fmt) looks it up for the current language and
# applies str.format(**fmt) when placeholders are present. Console/engine log lines are
# left untranslated (they are diagnostic and interleave with the engine's own output).
TR = {
    "es": {
        "lang_label": "Idioma / Language:",
        "tab_translator": "  Traductor  ",
        "tab_compare": "  Comparar / WinMerge  ",
        # status
        "status_idle": "Inactivo",
        "status_running": "Ejecutando…",
        "status_stopping": "Deteniendo…",
        "status_comparing": "Comparando…",
        "status_compare_ready": "Comparación lista ✔",
        "status_translating": "Traduciendo…",
        "status_autotranslate_ready": "Auto-traducción lista ✔",
        "status_done": "Hecho ✔",
        "status_done_warnings": "Hecho con avisos ⚠",
        "status_failed": "Falló ✖",
        # tab 1
        "frame_input_files": "Archivos de entrada (cola)",
        "col_xml_files": "Archivos XML (arrastra y suelta, o usa Añadir…)",
        "btn_add": "Añadir…",
        "btn_remove_selected": "Quitar seleccionados",
        "btn_clear": "Limpiar",
        "files_count": "{n} archivos",
        "dnd_hint": "(Instala 'tkinterdnd2' con pip para arrastrar y soltar.)",
        "frame_output_cache": "Salida y caché",
        "lbl_output_folder": "Carpeta de salida:",
        "btn_browse": "Examinar…",
        "lbl_cache_file": "Archivo de caché:",
        "btn_auto": "Auto",
        "hint_auto_paths": "(Deja los campos vacíos para rutas automáticas)",
        "frame_settings": "Ajustes",
        "lbl_source_language": "Traducir de:",
        "lbl_target_language": "a:",
        "lbl_api_key": "Clave API de Gemini:",
        "btn_show_hide": "Mostrar/Ocultar",
        "btn_test": "Probar",
        "chk_cache_only": "Solo caché",
        "chk_retry_empty": "Reintentar caché vacía",
        "chk_verbose": "Detallado",
        "btn_estimate": "Estimar costo",
        "btn_open_folder": "Abrir carpeta de salida",
        "btn_translate": "Traducir",
        "btn_stop": "Detener",
        "frame_log": "Registro",
        "lbl_search": "Buscar:",
        "est_no_files": "No hay archivos en cola.",
        "est_result": "≈ {strings} strings · ~{tokens} tokens · ~${cost} USD",
        "est_with_cache": "≈ {pending} a traducir ({cached} ya en caché) · ~{tokens} tokens · ~${cost} USD",
        "est_error": "Error de estimación: {exc}",
        "progress_batch": "Lote {done}/{total}",
        "btn_cache_only": "Solo caché",
        "btn_advanced": "Avanzado",
        "tip_cache_only_btn": "Aplica SOLO el caché (0 API, sin Gemini); lo no cacheado queda en inglés.",
        "tip_advanced": "Muestra/oculta opciones poco usadas.",
        # tab 1 messageboxes
        "mb_busy_title": "Ocupado",
        "mb_busy_msg": "Ya hay una traducción en curso.",
        "mb_op_busy_msg": "Ya hay una operación en curso.",
        "mb_no_files_title": "Sin archivos",
        "mb_no_files_msg": "Añade al menos un archivo XML.",
        "mb_missing_file_title": "Falta archivo",
        "mb_missing_file_msg": "El archivo no existe:\n{p}",
        "mb_missing_key_title": "Falta clave API",
        "mb_missing_key_msg": "Introduce una clave API de Gemini, o activa 'Solo caché'.",
        "mb_lang_mismatch_title": "El idioma de origen no coincide",
        "mb_lang_mismatch_msg": ("El archivo parece estar en {detected}, pero elegiste {selected} "
                                 "como idioma de origen.\n\n¿Cambiar el origen a {detected}?\n\n"
                                 "Sí = usar {detected}   ·   No = mantener {selected}   ·   "
                                 "Cancelar = no traducir"),
        "mb_no_key_title": "Sin clave",
        "mb_no_key_msg": "Introduce primero una clave API.",
        "mb_apikey_title": "Clave API",
        "mb_apikey_ok": "✅ Clave válida. Gemini respondió correctamente.",
        "mb_apikey_notext": "⚠️ La llamada tuvo éxito pero no devolvió texto.",
        "mb_apikey_fail": "❌ Falló la prueba de la clave:\n\n{msg}",
        "mb_open_folder_err": "No se pudo abrir la carpeta:\n{exc}",
        "mb_error_title": "Error",
        "title_add_xml": "Añadir archivos XML",
        "title_select_output": "Selecciona la carpeta de salida",
        "title_select_cache": "Selecciona el archivo de caché",
        "ft_xml": "Archivos XML",
        "ft_all": "Todos los archivos",
        "ft_json": "JSON",
        "ft_cache_json": "Caché JSON",
        "ft_xml_json": "XML / JSON",
        # tab 2
        "frame_files_compare": "Archivos a comparar",
        "lbl_new_en": "Inglés NUEVO:",
        "lbl_old_en": "Inglés VIEJO:",
        "lbl_old_trans": "Traducción vieja:",
        "lbl_cache_optional": "Caché (opcional):",
        "btn_compare": "Comparar",
        "title_pick_new": "Selecciona el XML nuevo en inglés",
        "title_pick_old_en": "Selecciona el XML viejo en inglés",
        "title_pick_old_trans": "Selecciona el XML viejo ya traducido",
        "title_pick_cache": "Selecciona el archivo de caché .json",
        "lbl_show": "Mostrar:",
        "filt_diff": "Diferencias (inglés)",
        "filt_pending": "A revisar (faltantes)",
        "filt_format": "Revisar formato",
        "filt_changed": "Cambiados",
        "filt_new": "Nuevos",
        "filt_reused": "Reusados/traducidos",
        "filt_all": "Todos",
        "col_source": "Origen nuevo",
        "col_old_source": "Origen viejo",
        "col_translation": "Traducción",
        "col_status": "Estado",
        "diff_new": "NUEVO:",
        "diff_old": "VIEJO:",
        "diff_same": "Sin cambios en el inglés.",
        "frame_comparison": "Comparación",
        "frame_edit": "Editar traducción",
        "btn_apply_edit": "Aplicar edición",
        "hint_select_row": "Selecciona una fila para editarla.",
        "hint_locid": "_locID {loc}",
        "hint_draft": "Borrador (traducción vieja) — revísalo y aplica.",
        "hint_no_translation": "_locID {loc} (sin traducción)",
        "btn_autotranslate": "Auto-traducir faltantes (Gemini)",
        "btn_save_cache": "Guardar en caché",
        "btn_export": "Exportar XML adaptado",
        "cmp_counts": "{total} total · {pending} sin traducir · {note}",
        "cmp_note": "{matched} coinciden · mostrando {shown}",
        "cmp_note_cap": " (tope {cap}; afina filtro/búsqueda)",
        "st_edited": "editado",
        "st_reuse_cosmetic": "reusado · revisar formato",
        "st_kept": "conservado (sin traducir)",
        "st_new_translated": "nuevo (traducido)",
        "st_changed_translated": "cambiado (traducido)",
        "st_reused": "reusado",
        "st_changed_pending": "cambiado · falta",
        "st_new_pending": "nuevo · falta",
        "st_pending": "falta traducir",
        "mb_compare_first": "Primero pulsa 'Comparar'.",
        "mb_pick_file_msg": "Selecciona el archivo: {label}.",
        "mb_not_exist_msg": "No existe:\n{p}",
        "mb_nothing_title": "Nada que hacer",
        "mb_nothing_translate_title": "Nada que traducir",
        "mb_nothing_translate_msg": "No quedan strings sin traducir.",
        "mb_need_key_or_cache": ("Sin clave API solo se puede rellenar desde caché. "
                                 "Pon una clave API o selecciona un archivo de caché."),
        "mb_confirm_title": "Confirmar",
        "mb_confirm_autotranslate": ("Se traducirán ~{pending} strings con Gemini "
                                     "(puede consumir API). ¿Continuar?"),
        "mb_no_selection_title": "Sin selección",
        "mb_no_selection_msg": "Selecciona una fila primero.",
        "mb_nothing_save_title": "Nada que guardar",
        "mb_save_cache_as": "Guardar caché como…",
        "mb_cache_saved_title": "Caché guardada",
        "mb_cache_saved_msg": "Se escribieron {n} traducciones en:\n{path}",
        "mb_cache_save_err": "No se pudo guardar la caché:\n{exc}",
        "mb_nothing_export_title": "Nada que exportar",
        "mb_export_as": "Exportar XML adaptado como…",
        "mb_exported_title": "Exportado",
        "mb_exported_msg": "XML adaptado escrito:\n{out}\n\n{done} traducidos · {pending} aún en inglés.",
        "mb_export_err": "No se pudo exportar:\n{exc}",
        "files_count_one": "{n} archivo",
        "log_parsing": "🔎 Analizando archivos…",
        "log_cache_read_err": "⚠️  No se pudo leer la caché: {exc}",
        "log_merge_summary": ("🔗 _locID: {unchanged} iguales, {changed} cambiados, {new} nuevos. "
                              "Reuso seguro: {seeded}, desde caché: {cache}."),
        "log_autotranslate_summary": "📊 Auto-traducción · caché: {cache} · API: {api}",
        # Build-cache (no API)
        "btn_build_cache": "Generar caché (sin API)…",
        "title_build_pick_eng": "1/3 · Selecciona el XML en INGLÉS",
        "title_build_pick_es": "2/3 · Selecciona su XML TRADUCIDO",
        "title_build_save": "3/3 · Guardar el caché como…",
        "status_building": "Generando caché…",
        "status_build_done": "Caché generada ✔",
        "log_build_summary": "💾 Caché generada: {n} entradas escritas (0 API).",
        "mb_build_done_title": "Caché generada",
        "mb_build_done_msg": "Se escribieron {n} entradas en:\n{path}\n\n(sin usar Gemini)",
        "mb_build_err": "No se pudo generar la caché:\n{exc}",
        "mb_build_confirm": ("El XML en INGLÉS debe ser la MISMA versión de la que se hizo esta "
                             "traducción.\nSi son de versiones distintas, se guardarán traducciones "
                             "desfasadas para las cadenas que cambiaron.\n\n¿Continuar?"),
        # Tooltips
        "tip_lang": "Cambia el idioma de TODA la interfaz (no afecta el idioma de traducción).",
        "tip_input_files": "Archivos XML a traducir. Arrastra y suelta, o usa 'Añadir…'. Se procesan en cola.",
        "tip_add": "Añadir archivos XML a la cola.",
        "tip_remove": "Quitar de la cola los archivos seleccionados.",
        "tip_clear": "Vaciar la cola de archivos.",
        "tip_output_folder": "Carpeta donde se escriben los XML traducidos (como <nombre>_translated.xml). Vacío = junto al original.",
        "tip_cache_file": "Archivo de caché a usar (origen→destino). Reutiliza traducciones y ahorra API. Usa UN caché por par de idiomas. Vacío = uno automático por archivo y par.",
        "tip_cache_auto": "Volver al caché automático (<salida>.cache.json, separado por par de idiomas).",
        "tip_source_lang": "Idioma original del XML de entrada (ej. inglés, chino, japonés).",
        "tip_target_lang": "Idioma al que se traduce el texto (ej. español latino). Distinto del idioma de la interfaz.",
        "tip_api_key": "Tu clave de Google Gemini. Solo se necesita para cadenas que NO estén en el caché. No se guarda en disco.",
        "tip_test": "Hace una llamada mínima a Gemini para comprobar que la clave funciona.",
        "tip_cache_only": "Usa SOLO el caché; nunca llama a Gemini (0 costo). Lo no cacheado queda sin traducir.",
        "tip_retry_empty": "Reintenta las cadenas que quedaron marcadas como vacías/fallidas en el caché.",
        "tip_verbose": "Muestra registro detallado (depuración) en el panel inferior.",
        "tip_estimate": ("Calcula cadenas/tokens y costo aproximado sin traducir nada. Incluye entrada, "
                         "salida y el prompt repetido por lote; los reintentos pueden subirlo un poco."),
        "tip_open_folder": "Abre la carpeta de salida en el explorador (al terminar).",
        "tip_translate": "Inicia la traducción de la cola (usa caché; llama a Gemini solo para lo no cacheado).",
        "tip_stop": "Cancela la operación en curso.",
        "tip_build_cache": "Crea un caché SIN usar Gemini: empareja un XML en inglés con su XML ya traducido (por _locID) y guarda inglés→español.",
        "tip_cmp_new": "Versión NUEVA en inglés (la que vas a traducir).",
        "tip_cmp_old_en": "Versión VIEJA en inglés (para detectar qué cambió).",
        "tip_cmp_old_trans": "Traducción VIEJA (de la que se reutiliza el español).",
        "tip_cmp_cache": "Caché opcional: rellena el español de lo ya traducido al comparar.",
        "tip_cmp_compare": "Compara los archivos y clasifica cada cadena (igual / cambiada / nueva).",
        "tip_cmp_filter": "Filtra qué filas se muestran (por defecto, solo lo que cambió en inglés).",
        "tip_cmp_search": "Busca dentro de _locID / inglés / traducción.",
        "tip_cmp_apply": "Aplica el texto editado a la fila seleccionada.",
        "tip_cmp_autotranslate": "Traduce con Gemini las cadenas nuevas/cambiadas que falten (necesita clave API).",
        "tip_cmp_save_cache": "Genera/actualiza el caché (origen→destino) con lo reutilizado/editado. No usa API.",
        "tip_cmp_export": "Escribe el XML final adaptado (español donde lo hay, inglés en lo pendiente).",
    },
    "en": {
        "lang_label": "Idioma / Language:",
        "tab_translator": "  Translator  ",
        "tab_compare": "  Compare / WinMerge  ",
        "status_idle": "Idle",
        "status_running": "Running…",
        "status_stopping": "Stopping…",
        "status_comparing": "Comparing…",
        "status_compare_ready": "Comparison ready ✔",
        "status_translating": "Translating…",
        "status_autotranslate_ready": "Auto-translation done ✔",
        "status_done": "Done ✔",
        "status_done_warnings": "Done with warnings ⚠",
        "status_failed": "Failed ✖",
        "frame_input_files": "Input files (queue)",
        "col_xml_files": "XML files (drag & drop supported, or use Add…)",
        "btn_add": "Add…",
        "btn_remove_selected": "Remove selected",
        "btn_clear": "Clear",
        "files_count": "{n} files",
        "dnd_hint": "(Install 'tkinterdnd2' via pip to enable drag & drop.)",
        "frame_output_cache": "Output & cache",
        "lbl_output_folder": "Output folder:",
        "btn_browse": "Browse…",
        "lbl_cache_file": "Cache file:",
        "btn_auto": "Auto",
        "hint_auto_paths": "(Leave fields empty for automatic paths)",
        "frame_settings": "Settings",
        "lbl_source_language": "Translate from:",
        "lbl_target_language": "to:",
        "lbl_api_key": "Gemini API key:",
        "btn_show_hide": "Show/Hide",
        "btn_test": "Test",
        "chk_cache_only": "Cache only",
        "chk_retry_empty": "Retry empty cache",
        "chk_verbose": "Verbose",
        "btn_estimate": "Estimate cost",
        "btn_open_folder": "Open output folder",
        "btn_translate": "Translate",
        "btn_stop": "Stop",
        "frame_log": "Log",
        "lbl_search": "Search:",
        "est_no_files": "No files queued.",
        "est_result": "≈ {strings} strings · ~{tokens} tokens · ~${cost} USD",
        "est_with_cache": "≈ {pending} to translate ({cached} already cached) · ~{tokens} tokens · ~${cost} USD",
        "est_error": "Estimate error: {exc}",
        "progress_batch": "Batch {done}/{total}",
        "btn_cache_only": "Cache only",
        "btn_advanced": "Advanced",
        "tip_cache_only_btn": "Apply the cache ONLY (no API, no Gemini); uncached strings stay in English.",
        "tip_advanced": "Show/hide rarely-used options.",
        "mb_busy_title": "Busy",
        "mb_busy_msg": "A translation is already running.",
        "mb_op_busy_msg": "An operation is already running.",
        "mb_no_files_title": "No files",
        "mb_no_files_msg": "Please add at least one XML file.",
        "mb_missing_file_title": "Missing file",
        "mb_missing_file_msg": "File does not exist:\n{p}",
        "mb_missing_key_title": "Missing API key",
        "mb_missing_key_msg": "Enter a Gemini API key, or enable 'Cache only'.",
        "mb_lang_mismatch_title": "Source language mismatch",
        "mb_lang_mismatch_msg": ("The file looks like it is in {detected}, but you selected "
                                 "{selected} as the source language.\n\nSwitch the source to "
                                 "{detected}?\n\nYes = use {detected}   ·   No = keep {selected}"
                                 "   ·   Cancel = abort"),
        "mb_no_key_title": "No key",
        "mb_no_key_msg": "Please enter an API key first.",
        "mb_apikey_title": "API key",
        "mb_apikey_ok": "✅ Key valid. Gemini responded successfully.",
        "mb_apikey_notext": "⚠️ The call succeeded but returned no text.",
        "mb_apikey_fail": "❌ Key test failed:\n\n{msg}",
        "mb_open_folder_err": "Could not open folder:\n{exc}",
        "mb_error_title": "Error",
        "title_add_xml": "Add XML files",
        "title_select_output": "Select output folder",
        "title_select_cache": "Select cache file",
        "ft_xml": "XML files",
        "ft_all": "All files",
        "ft_json": "JSON",
        "ft_cache_json": "Cache JSON",
        "ft_xml_json": "XML / JSON",
        "frame_files_compare": "Files to compare",
        "lbl_new_en": "New English:",
        "lbl_old_en": "Old English:",
        "lbl_old_trans": "Old translation:",
        "lbl_cache_optional": "Cache (optional):",
        "btn_compare": "Compare",
        "title_pick_new": "Select the new English XML",
        "title_pick_old_en": "Select the old English XML",
        "title_pick_old_trans": "Select the old translated XML",
        "title_pick_cache": "Select the .json cache file",
        "lbl_show": "Show:",
        "filt_diff": "Differences (English)",
        "filt_pending": "Needs review (missing)",
        "filt_format": "Review formatting",
        "filt_changed": "Changed",
        "filt_new": "New",
        "filt_reused": "Reused/translated",
        "filt_all": "All",
        "col_source": "New source",
        "col_old_source": "Old source",
        "col_translation": "Translation",
        "col_status": "Status",
        "diff_new": "NEW:",
        "diff_old": "OLD:",
        "diff_same": "No change in the English.",
        "frame_comparison": "Comparison",
        "frame_edit": "Edit translation",
        "btn_apply_edit": "Apply edit",
        "hint_select_row": "Select a row to edit it.",
        "hint_locid": "_locID {loc}",
        "hint_draft": "Draft (old translation) — review and apply.",
        "hint_no_translation": "_locID {loc} (no translation)",
        "btn_autotranslate": "Auto-translate missing (Gemini)",
        "btn_save_cache": "Save to cache",
        "btn_export": "Export adapted XML",
        "cmp_counts": "{total} total · {pending} untranslated · {note}",
        "cmp_note": "{matched} match · showing {shown}",
        "cmp_note_cap": " (cap {cap}; refine filter/search)",
        "st_edited": "edited",
        "st_reuse_cosmetic": "reused · review formatting",
        "st_kept": "kept (untranslated)",
        "st_new_translated": "new (translated)",
        "st_changed_translated": "changed (translated)",
        "st_reused": "reused",
        "st_changed_pending": "changed · missing",
        "st_new_pending": "new · missing",
        "st_pending": "to translate",
        "mb_compare_first": "Click 'Compare' first.",
        "mb_pick_file_msg": "Select the file: {label}.",
        "mb_not_exist_msg": "Does not exist:\n{p}",
        "mb_nothing_title": "Nothing to do",
        "mb_nothing_translate_title": "Nothing to translate",
        "mb_nothing_translate_msg": "No untranslated strings left.",
        "mb_need_key_or_cache": ("Without an API key you can only fill from cache. "
                                 "Add an API key or select a cache file."),
        "mb_confirm_title": "Confirm",
        "mb_confirm_autotranslate": ("~{pending} strings will be translated with Gemini "
                                     "(may use API). Continue?"),
        "mb_no_selection_title": "No selection",
        "mb_no_selection_msg": "Select a row first.",
        "mb_nothing_save_title": "Nothing to save",
        "mb_save_cache_as": "Save cache as…",
        "mb_cache_saved_title": "Cache saved",
        "mb_cache_saved_msg": "Wrote {n} translations to:\n{path}",
        "mb_cache_save_err": "Could not save the cache:\n{exc}",
        "mb_nothing_export_title": "Nothing to export",
        "mb_export_as": "Export adapted XML as…",
        "mb_exported_title": "Exported",
        "mb_exported_msg": "Adapted XML written:\n{out}\n\n{done} translated · {pending} still in English.",
        "mb_export_err": "Could not export:\n{exc}",
        "files_count_one": "{n} file",
        "log_parsing": "🔎 Parsing files…",
        "log_cache_read_err": "⚠️  Could not read the cache: {exc}",
        "log_merge_summary": ("🔗 _locID: {unchanged} unchanged, {changed} changed, {new} new. "
                              "Safe reuse: {seeded}, from cache: {cache}."),
        "log_autotranslate_summary": "📊 Auto-translation · cache: {cache} · API: {api}",
        # Build-cache (no API)
        "btn_build_cache": "Build cache (no API)…",
        "title_build_pick_eng": "1/3 · Select the ENGLISH XML",
        "title_build_pick_es": "2/3 · Select its TRANSLATED XML",
        "title_build_save": "3/3 · Save the cache as…",
        "status_building": "Building cache…",
        "status_build_done": "Cache built ✔",
        "log_build_summary": "💾 Cache built: {n} entries written (0 API).",
        "mb_build_done_title": "Cache built",
        "mb_build_done_msg": "Wrote {n} entries to:\n{path}\n\n(no Gemini used)",
        "mb_build_err": "Could not build the cache:\n{exc}",
        "mb_build_confirm": ("The ENGLISH XML must be the SAME version this translation was made "
                             "from.\nIf they are different versions, outdated translations will be "
                             "cached for changed strings.\n\nContinue?"),
        # Tooltips
        "tip_lang": "Switches the language of the WHOLE interface (not the translation target).",
        "tip_input_files": "XML files to translate. Drag & drop, or use 'Add…'. Processed as a queue.",
        "tip_add": "Add XML files to the queue.",
        "tip_remove": "Remove the selected files from the queue.",
        "tip_clear": "Empty the file queue.",
        "tip_output_folder": "Folder where translated XML is written (as <name>_translated.xml). Empty = next to the original.",
        "tip_cache_file": "Cache file to use (source→target). Reuses translations and saves API. Use ONE cache per language pair. Empty = one automatic per file and pair.",
        "tip_cache_auto": "Revert to the automatic cache (<output>.cache.json, split per language pair).",
        "tip_source_lang": "The original language of the input XML (e.g. English, Chinese, Japanese).",
        "tip_target_lang": "The language the text is translated into (e.g. Latin American Spanish). Separate from the UI language.",
        "tip_api_key": "Your Google Gemini key. Needed only for strings NOT in the cache. Not saved to disk.",
        "tip_test": "Makes a tiny Gemini call to check the key works.",
        "tip_cache_only": "Use the cache ONLY; never call Gemini (zero cost). Uncached strings stay untranslated.",
        "tip_retry_empty": "Retry strings previously cached as empty/failed.",
        "tip_verbose": "Show detailed (debug) logging in the panel below.",
        "tip_estimate": ("Estimate strings/tokens and a rough cost without translating. Covers input, "
                         "output and the per-batch prompt overhead; retries may add a little."),
        "tip_open_folder": "Open the output folder in the file explorer (when finished).",
        "tip_translate": "Start translating the queue (uses cache; calls Gemini only for uncached strings).",
        "tip_stop": "Cancel the running operation.",
        "tip_build_cache": "Build a cache WITHOUT Gemini: pair an English XML with its already-translated XML (by _locID) and store English→Spanish.",
        "tip_cmp_new": "NEW English version (the one you will translate).",
        "tip_cmp_old_en": "OLD English version (to detect what changed).",
        "tip_cmp_old_trans": "OLD translation (the Spanish to reuse).",
        "tip_cmp_cache": "Optional cache: fills in the Spanish of already-translated strings on compare.",
        "tip_cmp_compare": "Compare the files and classify each string (unchanged / changed / new).",
        "tip_cmp_filter": "Filter which rows show (by default, only what changed in English).",
        "tip_cmp_search": "Search within _locID / English / translation.",
        "tip_cmp_apply": "Apply the edited text to the selected row.",
        "tip_cmp_autotranslate": "Translate the missing new/changed strings with Gemini (needs an API key).",
        "tip_cmp_save_cache": "Build/update the cache (source→target) from reused/edited strings. No API.",
        "tip_cmp_export": "Write the final adapted XML (Spanish where available, English for pending).",
    },
}
# Filter dropdown: stable code + translation key, in display order.
CMP_FILTER_CODES = [
    ("diff", "filt_diff"),
    ("pending", "filt_pending"),
    ("format", "filt_format"),
    ("changed", "filt_changed"),
    ("new", "filt_new"),
    ("reused", "filt_reused"),
    ("all", "filt_all"),
]


def _enable_windows_dpi_awareness() -> None:
    """Tell Windows we can draw at native DPI so it doesn't bitmap-scale the window.

    Without this, Tk windows on High-DPI screens (HD, 4K, scaled displays) are
    bitmap-stretched by Windows, producing blurry text. Calling this API once
    before creating the Tk root makes the UI crisp.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE (value 2) gives us per-monitor scaling.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except (AttributeError, OSError):
            pass
        # Fallback for older Windows versions.
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _apply_tk_scaling(root: tk.Tk) -> float:
    """Compute a Tk scaling factor that matches the monitor's DPI.

    Tk uses 72 DPI as reference. On a 150% (144 DPI) display, we want Tk scaling = 2.0
    so fonts and widgets render at true pixel resolution.
    """
    try:
        # winfo_fpixels("1i") returns how many pixels equal one inch on the current display.
        dpi = root.winfo_fpixels("1i")
        # Clamp to a sensible range to avoid absurd scaling on buggy drivers.
        scaling = max(1.0, min(dpi / 72.0, 3.0))
        root.tk.call("tk", "scaling", scaling)
        return scaling
    except Exception:
        return 1.0


class QueueWriter:
    """File-like that pushes writes into a thread-safe queue."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        self.q = q

    def write(self, text: str) -> None:
        if text:
            self.q.put(text)

    def flush(self) -> None:
        pass


class Tooltip:
    """A lightweight hover tooltip for any Tk widget.

    `text_provider` is a callable returning the current text, so the tooltip follows the
    selected UI language (it is read at hover time, not at creation).
    """

    def __init__(self, widget: tk.Misc, text_provider) -> None:
        self.widget = widget
        self.text_provider = text_provider
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self._tip is not None:
            return
        text = self.text_provider() if callable(self.text_provider) else self.text_provider
        if not text:
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=text, justify="left",
            background="#3a3a3a", foreground="#f0f0f0", relief="solid", borderwidth=1,
            wraplength=420, padx=7, pady=5,
            font=("Segoe UI", 9) if sys.platform == "win32" else ("Helvetica", 10),
        ).pack()

    def _hide(self, _event=None) -> None:
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class TranslatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(860, 720)
        self.root.configure(bg=BG_MAIN)

        self._init_state()
        self._load_config()
        self._apply_dark_theme()

        # Persistent top bar with the language selector (survives UI rebuilds).
        topbar = ttk.Frame(self.root)
        topbar.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(topbar, text=self.t("lang_label"), style="Muted.TLabel").pack(side="left")
        self.lang_combo = ttk.Combobox(
            topbar, width=12, state="readonly",
            values=[name for name, _code in UI_LANGUAGES],
        )
        self.lang_combo.set(self._lang_display(self.lang.get()))
        self.lang_combo.pack(side="left", padx=(6, 0))
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)
        self._tip(self.lang_combo, "tip_lang")

        # The notebook lives in a container so we can rebuild it on a language switch.
        self.notebook_container = ttk.Frame(self.root)
        self.notebook_container.pack(fill="both", expand=True)
        self.notebook = None
        self._build_layout()

        self.root.after(80, self._drain_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------- state & i18n --

    def _init_state(self) -> None:
        """Create every persistent var once, so a language rebuild keeps user input/data."""
        self.lang = tk.StringVar(value="es")
        self.input_paths: List[Path] = []
        self.output_path = tk.StringVar()
        self.api_key = tk.StringVar(
            value=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        )
        self.source_lang = tk.StringVar(value=tg.DEFAULT_SOURCE_LANG)
        self.target_lang = tk.StringVar(value=tg.DEFAULT_TARGET_LANG)
        self.cache_only = tk.BooleanVar(value=False)
        self.retry_empty = tk.BooleanVar(value=False)
        self.verbose = tk.BooleanVar(value=False)
        self.cache_file_path = tk.StringVar()
        self.search_text = tk.StringVar()

        self.last_output_path: Optional[Path] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.cancel_event: Optional[threading.Event] = None
        self._full_log = ""

        # Compare / WinMerge tab state.
        self.cmp_new_path = tk.StringVar()
        self.cmp_old_source_path = tk.StringVar()
        self.cmp_old_trans_path = tk.StringVar()
        self.cmp_cache_path = tk.StringVar()
        self.cmp_filter_code = "diff"
        self.cmp_search = tk.StringVar()
        # Traces added once here (vars persist across UI rebuilds, widgets don't).
        self.search_text.trace_add("write", lambda *_: self._apply_search_filter())
        self.cmp_search.trace_add("write", lambda *_: self._cmp_populate_tree())
        self.cmp_entries: List = []
        self.cmp_values: List[str] = []
        self.cmp_edited: set = set()
        self.cmp_new_targets: List = []
        self.cmp_all_targets: List = []
        self.cmp_tree = None
        self.cmp_doc_format = None
        self.cmp_elements: List = []

    def t(self, key: str, **fmt) -> str:
        """Translate `key` for the current UI language; apply str.format(**fmt) if given."""
        table = TR.get(self.lang.get(), TR["es"])
        text = table.get(key) or TR["es"].get(key) or key
        return text.format(**fmt) if fmt else text

    def _tip(self, widget: tk.Misc, key: str) -> tk.Misc:
        """Attach a language-aware hover tooltip (reads self.t(key) at hover time)."""
        Tooltip(widget, lambda: self.t(key))
        return widget

    @staticmethod
    def _lang_display(code: str) -> str:
        for name, c in UI_LANGUAGES:
            if c == code:
                return name
        return UI_LANGUAGES[0][0]

    @staticmethod
    def _lang_code(display: str) -> str:
        for name, c in UI_LANGUAGES:
            if name == display:
                return c
        return "es"

    def _on_language_change(self, _event=None) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(self.t("mb_busy_title"), self.t("mb_op_busy_msg"))
            self.lang_combo.set(self._lang_display(self.lang.get()))
            return
        self.lang.set(self._lang_code(self.lang_combo.get()))
        self._save_config()
        self._build_layout()

    def _build_layout(self) -> None:
        """(Re)build the tabbed UI in the current language, preserving state."""
        if self.notebook is not None:
            self.notebook.destroy()
        self.notebook = ttk.Notebook(self.notebook_container)
        self.notebook.pack(fill="both", expand=True)
        translate_tab = ttk.Frame(self.notebook)
        compare_tab = ttk.Frame(self.notebook)
        self.notebook.add(translate_tab, text=self.t("tab_translator"))
        self.notebook.add(compare_tab, text=self.t("tab_compare"))
        self._build_ui(translate_tab)
        self._build_compare_ui(compare_tab)

        # Restore the console log and the comparison table into the fresh widgets.
        if self._full_log:
            self.console.configure(state="normal")
            self.console.insert("end", self._full_log)
            self.console.configure(state="disabled")
            self._tag_log_lines()
            self.console.see("end")
        if self.cmp_entries:
            self._cmp_populate_tree()

    # ------------------------------------------------------------- theming --

    def _apply_dark_theme(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG_MAIN, foreground=FG_MAIN, fieldbackground=BG_ENTRY)
        style.configure("TFrame", background=BG_MAIN)
        style.configure("TLabel", background=BG_MAIN, foreground=FG_MAIN)
        style.configure("Muted.TLabel", background=BG_MAIN, foreground=FG_MUTED)
        style.configure("Accent.TLabel", background=BG_MAIN, foreground=FG_ACCENT)
        style.configure("TLabelframe", background=BG_FRAME, foreground=FG_ACCENT, borderwidth=1)
        style.configure("TLabelframe.Label", background=BG_MAIN, foreground=FG_ACCENT)
        style.configure("TEntry", fieldbackground=BG_ENTRY, foreground=FG_MAIN,
                        insertcolor=FG_MAIN, bordercolor=BG_FRAME)
        style.configure("TButton", background=BG_ENTRY, foreground=FG_MAIN,
                        bordercolor=BG_FRAME, focusthickness=0, padding=6)
        style.map("TButton",
                  background=[("active", "#3a3a3a"), ("pressed", "#454545")])
        style.configure("Accent.TButton", background=FG_ACCENT, foreground="#000000", padding=8)
        style.map("Accent.TButton",
                  background=[("active", "#5fd9bf"), ("disabled", "#555555")])
        style.configure("TCheckbutton", background=BG_MAIN, foreground=FG_MAIN)
        style.map("TCheckbutton",
                  background=[("active", BG_MAIN)], foreground=[("active", FG_MAIN)])
        style.configure("TCombobox", fieldbackground=BG_ENTRY, background=BG_ENTRY,
                        foreground=FG_MAIN, arrowcolor=FG_MAIN, bordercolor=BG_FRAME)
        style.map("TCombobox", fieldbackground=[("readonly", BG_ENTRY)])
        style.configure("Horizontal.TProgressbar", background=FG_ACCENT,
                        troughcolor=BG_ENTRY, bordercolor=BG_FRAME,
                        lightcolor=FG_ACCENT, darkcolor=FG_ACCENT)
        style.configure("Treeview", background=BG_ENTRY, foreground=FG_MAIN,
                        fieldbackground=BG_ENTRY, borderwidth=0)
        style.configure("Treeview.Heading", background=BG_FRAME, foreground=FG_ACCENT,
                        borderwidth=0)
        style.map("Treeview", background=[("selected", "#264f78")])
        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_FRAME, foreground=FG_MAIN,
                        padding=(14, 7), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_ENTRY)],
                  foreground=[("selected", FG_ACCENT)])

        self.root.option_add("*TCombobox*Listbox*Background", BG_ENTRY)
        self.root.option_add("*TCombobox*Listbox*Foreground", FG_MAIN)
        self.root.option_add("*TCombobox*Listbox*selectBackground", "#264f78")

    # ------------------------------------------------------------------ UI --

    def _build_ui(self, parent: tk.Misc) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- File queue -----------------------------------------------------
        files_frame = ttk.LabelFrame(parent, text=self.t("frame_input_files"), padding=8)
        files_frame.pack(fill="x", padx=10, pady=(10, 4))

        list_container = ttk.Frame(files_frame)
        list_container.pack(fill="x")

        self.files_tree = ttk.Treeview(
            list_container, columns=("path",), show="headings",
            height=4, selectmode="extended",
        )
        self.files_tree.heading("path", text=self.t("col_xml_files"))
        self.files_tree.column("path", anchor="w")
        self.files_tree.pack(side="left", fill="x", expand=True)
        self._tip(self.files_tree, "tip_input_files")

        scroll = ttk.Scrollbar(list_container, orient="vertical", command=self.files_tree.yview)
        scroll.pack(side="right", fill="y")
        self.files_tree.configure(yscrollcommand=scroll.set)

        file_buttons = ttk.Frame(files_frame)
        file_buttons.pack(fill="x", pady=(6, 0))
        _b = ttk.Button(file_buttons, text=self.t("btn_add"), command=self._add_files)
        _b.pack(side="left"); self._tip(_b, "tip_add")
        _b = ttk.Button(file_buttons, text=self.t("btn_remove_selected"), command=self._remove_selected_files)
        _b.pack(side="left", padx=4); self._tip(_b, "tip_remove")
        _b = ttk.Button(file_buttons, text=self.t("btn_clear"), command=self._clear_files)
        _b.pack(side="left"); self._tip(_b, "tip_clear")

        self.file_count_label = ttk.Label(file_buttons, text=self.t("files_count", n=0),
                                           style="Muted.TLabel")
        self.file_count_label.pack(side="right")

        if DND_AVAILABLE:
            self.files_tree.drop_target_register(DND_FILES)
            self.files_tree.dnd_bind("<<Drop>>", self._on_drop)
        else:
            ttk.Label(
                files_frame,
                text=self.t("dnd_hint"),
                style="Muted.TLabel",
            ).pack(anchor="w", pady=(4, 0))

        # --- Output + cache -------------------------------------------------
        io_frame = ttk.LabelFrame(parent, text=self.t("frame_output_cache"), padding=8)
        io_frame.pack(fill="x", padx=10, pady=4)

        _lbl = ttk.Label(io_frame, text=self.t("lbl_output_folder"))
        _lbl.grid(row=0, column=0, sticky="w", **pad); self._tip(_lbl, "tip_output_folder")
        _e = ttk.Entry(io_frame, textvariable=self.output_path)
        _e.grid(row=0, column=1, sticky="ew", **pad); self._tip(_e, "tip_output_folder")
        ttk.Button(io_frame, text=self.t("btn_browse"),
                   command=self._browse_output_folder).grid(row=0, column=2, **pad)

        _lbl = ttk.Label(io_frame, text=self.t("lbl_cache_file"))
        _lbl.grid(row=1, column=0, sticky="w", **pad); self._tip(_lbl, "tip_cache_file")
        _e = ttk.Entry(io_frame, textvariable=self.cache_file_path)
        _e.grid(row=1, column=1, sticky="ew", **pad); self._tip(_e, "tip_cache_file")
        cache_buttons = ttk.Frame(io_frame)
        cache_buttons.grid(row=1, column=2, **pad)
        ttk.Button(cache_buttons, text=self.t("btn_browse"), command=self._browse_cache).pack(side="left")
        _b = ttk.Button(cache_buttons, text=self.t("btn_auto"), command=lambda: self.cache_file_path.set(""))
        _b.pack(side="left", padx=(4, 0)); self._tip(_b, "tip_cache_auto")

        ttk.Label(io_frame, text=self.t("hint_auto_paths"),
                  style="Muted.TLabel").grid(row=2, column=1, sticky="w", padx=8)
        io_frame.columnconfigure(1, weight=1)

        # --- Settings -------------------------------------------------------
        settings_frame = ttk.LabelFrame(parent, text=self.t("frame_settings"), padding=8)
        settings_frame.pack(fill="x", padx=10, pady=4)

        _lbl = ttk.Label(settings_frame, text=self.t("lbl_source_language"))
        _lbl.grid(row=0, column=0, sticky="w", **pad); self._tip(_lbl, "tip_source_lang")
        lang_row = ttk.Frame(settings_frame)
        lang_row.grid(row=0, column=1, sticky="w", **pad)
        _cb = ttk.Combobox(lang_row, textvariable=self.source_lang,
                           values=DEFAULT_SOURCE_OPTIONS, width=24)
        _cb.pack(side="left"); self._tip(_cb, "tip_source_lang")
        _lbl = ttk.Label(lang_row, text=self.t("lbl_target_language"))
        _lbl.pack(side="left", padx=(8, 8)); self._tip(_lbl, "tip_target_lang")
        _cb = ttk.Combobox(lang_row, textvariable=self.target_lang,
                           values=DEFAULT_TARGET_OPTIONS, width=24)
        _cb.pack(side="left"); self._tip(_cb, "tip_target_lang")

        _lbl = ttk.Label(settings_frame, text=self.t("lbl_api_key"))
        _lbl.grid(row=1, column=0, sticky="w", **pad); self._tip(_lbl, "tip_api_key")
        api_entry = ttk.Entry(settings_frame, textvariable=self.api_key, show="•")
        api_entry.grid(row=1, column=1, sticky="ew", **pad); self._tip(api_entry, "tip_api_key")
        btn_row = ttk.Frame(settings_frame)
        btn_row.grid(row=1, column=2, **pad)
        ttk.Button(btn_row, text=self.t("btn_show_hide"),
                   command=lambda: self._toggle_secret(api_entry)).pack(side="left")
        _b = ttk.Button(btn_row, text=self.t("btn_test"), command=self._test_api_key)
        _b.pack(side="left", padx=(4, 0)); self._tip(_b, "tip_test")

        # "Advanced" toggle hides the rarely-used options to keep the tab uncluttered.
        self._advanced_open = False
        self._advanced_btn = ttk.Button(settings_frame, command=self._toggle_advanced)
        self._advanced_btn.grid(row=2, column=0, sticky="w", padx=8, pady=(8, 2))
        self._tip(self._advanced_btn, "tip_advanced")
        self._advanced_frame = ttk.Frame(settings_frame)
        self._advanced_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 2))
        _c = ttk.Checkbutton(self._advanced_frame, text=self.t("chk_retry_empty"), variable=self.retry_empty)
        _c.pack(side="left", padx=(0, 16)); self._tip(_c, "tip_retry_empty")
        _c = ttk.Checkbutton(self._advanced_frame, text=self.t("chk_verbose"), variable=self.verbose)
        _c.pack(side="left"); self._tip(_c, "tip_verbose")
        self._advanced_frame.grid_remove()
        self._refresh_advanced_btn()

        settings_frame.columnconfigure(1, weight=1)

        # --- Action bar -----------------------------------------------------
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill="x", padx=10, pady=(6, 4))

        self.estimate_label = ttk.Label(action_frame, text="", style="Accent.TLabel")
        self.estimate_label.pack(side="left", padx=(0, 10))
        _b = ttk.Button(action_frame, text=self.t("btn_estimate"), command=self._estimate_cost)
        _b.pack(side="left"); self._tip(_b, "tip_estimate")

        self.open_folder_button = ttk.Button(
            action_frame, text=self.t("btn_open_folder"),
            command=self._open_output_folder, state="disabled")
        self.open_folder_button.pack(side="right")
        self._tip(self.open_folder_button, "tip_open_folder")

        # --- Run / stop / progress ------------------------------------------
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", padx=10, pady=4)

        self.run_button = ttk.Button(run_frame, text=self.t("btn_translate"),
                                     style="Accent.TButton", command=self._on_run_clicked)
        self.run_button.pack(side="left")
        self._tip(self.run_button, "tip_translate")

        self.cache_only_btn = ttk.Button(run_frame, text=self.t("btn_cache_only"),
                                         command=self._on_cache_only_clicked)
        self.cache_only_btn.pack(side="left", padx=(6, 0))
        self._tip(self.cache_only_btn, "tip_cache_only_btn")

        self.stop_button = ttk.Button(run_frame, text=self.t("btn_stop"),
                                      command=self._on_stop_clicked, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        self._tip(self.stop_button, "tip_stop")

        self.progress = ttk.Progressbar(run_frame, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        self.status_label = ttk.Label(run_frame, text=self.t("status_idle"), style="Muted.TLabel")
        self.status_label.pack(side="right")

        # --- Log console with search ----------------------------------------
        log_frame = ttk.LabelFrame(parent, text=self.t("frame_log"), padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        search_row = ttk.Frame(log_frame)
        search_row.pack(fill="x", pady=(0, 4))
        ttk.Label(search_row, text=self.t("lbl_search")).pack(side="left", padx=(0, 4))
        ttk.Entry(search_row, textvariable=self.search_text).pack(
            side="left", fill="x", expand=True)
        ttk.Button(search_row, text=self.t("btn_clear"), command=self._clear_log).pack(side="right")

        self.console = scrolledtext.ScrolledText(
            log_frame, wrap="word", height=12,
            font=("Consolas", 11) if sys.platform == "win32" else ("Menlo", 12),
            state="disabled",
            background=BG_LOG, foreground=FG_MAIN, insertbackground=FG_MAIN,
            borderwidth=0, highlightthickness=0,
        )
        self.console.pack(fill="both", expand=True)
        self.console.tag_configure("match", background="#3a3a00", foreground="#ffff00")
        self.console.tag_configure("warn", foreground=FG_WARN)
        self.console.tag_configure("error", foreground=FG_ERROR)
        self.console.tag_configure("success", foreground=FG_SUCCESS)

        self._refresh_files_tree()

    # =====================================================================
    #  Compare / WinMerge tab
    # =====================================================================

    # How many rows we are willing to draw at once (Treeview gets slow past this).
    CMP_MAX_ROWS = 3000

    def _filter_label_for_code(self, code: str) -> str:
        for c, key in CMP_FILTER_CODES:
            if c == code:
                return self.t(key)
        return self.t(CMP_FILTER_CODES[0][1])

    def _on_cmp_filter_change(self) -> None:
        label = self.cmp_filter.get()
        if label in self._filter_labels:
            self.cmp_filter_code = CMP_FILTER_CODES[self._filter_labels.index(label)][0]
        self._cmp_populate_tree()

    def _build_compare_ui(self, parent: tk.Misc) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- File selectors -------------------------------------------------
        files = ttk.LabelFrame(parent, text=self.t("frame_files_compare"), padding=8)
        files.pack(fill="x", padx=10, pady=(10, 4))
        files.columnconfigure(1, weight=1)

        def _row(r: int, label_key: str, var: tk.StringVar, title_key: str, tip_key: str) -> None:
            _l = ttk.Label(files, text=self.t(label_key))
            _l.grid(row=r, column=0, sticky="w", **pad); self._tip(_l, tip_key)
            _e = ttk.Entry(files, textvariable=var)
            _e.grid(row=r, column=1, sticky="ew", **pad); self._tip(_e, tip_key)
            ttk.Button(files, text=self.t("btn_browse"),
                       command=lambda: self._cmp_browse(var, self.t(title_key))).grid(row=r, column=2, **pad)

        _row(0, "lbl_new_en", self.cmp_new_path, "title_pick_new", "tip_cmp_new")
        _row(1, "lbl_old_en", self.cmp_old_source_path, "title_pick_old_en", "tip_cmp_old_en")
        _row(2, "lbl_old_trans", self.cmp_old_trans_path, "title_pick_old_trans", "tip_cmp_old_trans")
        _l = ttk.Label(files, text=self.t("lbl_cache_optional"))
        _l.grid(row=3, column=0, sticky="w", **pad); self._tip(_l, "tip_cmp_cache")
        _e = ttk.Entry(files, textvariable=self.cmp_cache_path)
        _e.grid(row=3, column=1, sticky="ew", **pad); self._tip(_e, "tip_cmp_cache")
        cache_btns = ttk.Frame(files)
        cache_btns.grid(row=3, column=2, **pad)
        ttk.Button(cache_btns, text=self.t("btn_browse"),
                   command=lambda: self._cmp_browse(self.cmp_cache_path,
                                                    self.t("title_pick_cache"))).pack(side="left")
        _b = ttk.Button(files, text=self.t("btn_compare"), style="Accent.TButton",
                        command=self._on_compare_clicked)
        _b.grid(row=4, column=1, sticky="w", pady=(8, 2)); self._tip(_b, "tip_cmp_compare")

        # --- Filter + search ------------------------------------------------
        filt = ttk.Frame(parent)
        filt.pack(fill="x", padx=10, pady=2)
        ttk.Label(filt, text=self.t("lbl_show")).pack(side="left", padx=(0, 4))
        self._filter_labels = [self.t(key) for _code, key in CMP_FILTER_CODES]
        self.cmp_filter = tk.StringVar(value=self._filter_label_for_code(self.cmp_filter_code))
        _fc = ttk.Combobox(filt, textvariable=self.cmp_filter, width=24, state="readonly",
                           values=self._filter_labels)
        _fc.pack(side="left"); self._tip(_fc, "tip_cmp_filter")
        self.cmp_filter.trace_add("write", lambda *_: self._on_cmp_filter_change())
        ttk.Label(filt, text=self.t("lbl_search")).pack(side="left", padx=(12, 4))
        _se = ttk.Entry(filt, textvariable=self.cmp_search)
        _se.pack(side="left", fill="x", expand=True); self._tip(_se, "tip_cmp_search")
        self.cmp_counts_label = ttk.Label(filt, text="", style="Muted.TLabel")
        self.cmp_counts_label.pack(side="right")

        # --- Comparison table ----------------------------------------------
        table = ttk.LabelFrame(parent, text=self.t("frame_comparison"), padding=6)
        table.pack(fill="both", expand=True, padx=10, pady=4)

        cols = ("locid", "source", "oldsource", "translation", "status")
        self.cmp_treeview = ttk.Treeview(table, columns=cols, show="headings", selectmode="browse")
        for key, title, width in (
            ("locid", "_locID", 70),
            ("source", self.t("col_source"), 300),
            ("oldsource", self.t("col_old_source"), 300),
            ("translation", self.t("col_translation"), 300),
            ("status", self.t("col_status"), 140),
        ):
            self.cmp_treeview.heading(key, text=title)
            self.cmp_treeview.column(key, width=width, anchor="w",
                                     stretch=(key in ("source", "oldsource", "translation")))
        self.cmp_treeview.pack(side="left", fill="both", expand=True)
        tvscroll = ttk.Scrollbar(table, orient="vertical", command=self.cmp_treeview.yview)
        tvscroll.pack(side="right", fill="y")
        self.cmp_treeview.configure(yscrollcommand=tvscroll.set)
        self.cmp_treeview.tag_configure("pending", foreground=FG_ERROR)
        self.cmp_treeview.tag_configure("changed", foreground=FG_WARN)
        self.cmp_treeview.tag_configure("new", foreground=FG_ACCENT)
        self.cmp_treeview.tag_configure("ok", foreground=FG_SUCCESS)
        self.cmp_treeview.bind("<<TreeviewSelect>>", self._on_cmp_select)

        # --- Editor ---------------------------------------------------------
        editor = ttk.LabelFrame(parent, text=self.t("frame_edit"), padding=6)
        editor.pack(fill="x", padx=10, pady=4)
        # Read-only new-vs-old English diff with per-word highlighting (WinMerge-style).
        _mono = ("Consolas", 11) if sys.platform == "win32" else ("Menlo", 12)
        self.cmp_diff = scrolledtext.ScrolledText(
            editor, wrap="word", height=4, font=_mono, state="disabled",
            background=BG_LOG, foreground=FG_MAIN, borderwidth=0, highlightthickness=0,
        )
        self.cmp_diff.pack(fill="x", expand=True, pady=(0, 4))
        self.cmp_diff.tag_configure("ins", background="#143a14", foreground=FG_SUCCESS)
        self.cmp_diff.tag_configure("del", background="#3a1414", foreground=FG_ERROR)
        self.cmp_diff.tag_configure("label", foreground=FG_ACCENT)
        # Editable Spanish translation for the selected row.
        self.cmp_editor = scrolledtext.ScrolledText(
            editor, wrap="word", height=3, font=_mono,
            background=BG_LOG, foreground=FG_MAIN, insertbackground=FG_MAIN,
            borderwidth=0, highlightthickness=0,
        )
        self.cmp_editor.pack(fill="x", expand=True, pady=(0, 4))
        _b = ttk.Button(editor, text=self.t("btn_apply_edit"), command=self._on_cmp_apply_edit)
        _b.pack(side="left"); self._tip(_b, "tip_cmp_apply")
        self.cmp_editor_hint = ttk.Label(editor, text=self.t("hint_select_row"),
                                         style="Muted.TLabel")
        self.cmp_editor_hint.pack(side="left", padx=10)

        # --- Action buttons -------------------------------------------------
        actions = ttk.Frame(parent)
        actions.pack(fill="x", padx=10, pady=(2, 10))
        self.cmp_autotranslate_btn = ttk.Button(
            actions, text=self.t("btn_autotranslate"), command=self._on_cmp_autotranslate)
        self.cmp_autotranslate_btn.pack(side="left")
        self._tip(self.cmp_autotranslate_btn, "tip_cmp_autotranslate")
        _b = ttk.Button(actions, text=self.t("btn_save_cache"), command=self._on_cmp_save_cache)
        _b.pack(side="left", padx=6); self._tip(_b, "tip_cmp_save_cache")
        _b = ttk.Button(actions, text=self.t("btn_export"), command=self._on_cmp_export)
        _b.pack(side="left"); self._tip(_b, "tip_cmp_export")
        # Standalone no-API cache builder (English + its translation → cache), lives here now.
        _b = ttk.Button(actions, text=self.t("btn_build_cache"), command=self._on_build_cache)
        _b.pack(side="right"); self._tip(_b, "tip_build_cache")

    def _cmp_browse(self, var: tk.StringVar, title: str) -> None:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[(self.t("ft_xml_json"), "*.xml *.json"), (self.t("ft_all"), "*.*")],
        )
        if path:
            var.set(path)

    # ---------------------------------------------------- compare: run --

    def _on_compare_clicked(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(self.t("mb_busy_title"), self.t("mb_op_busy_msg"))
            return
        for label_key, var in (("lbl_new_en", self.cmp_new_path),
                               ("lbl_old_en", self.cmp_old_source_path),
                               ("lbl_old_trans", self.cmp_old_trans_path)):
            p = var.get().strip()
            if not p:
                messagebox.showerror(self.t("mb_missing_file_title"),
                                     self.t("mb_pick_file_msg", label=self.t(label_key)))
                return
            if not Path(p).exists():
                messagebox.showerror(self.t("mb_missing_file_title"), self.t("mb_not_exist_msg", p=p))
                return

        self._clear_log()
        self.cancel_event = threading.Event()
        self.status_label.configure(text=self.t("status_comparing"), foreground=FG_ACCENT)
        self.progress.configure(value=0, maximum=100)
        self.worker = threading.Thread(target=self._run_compare, daemon=True)
        self.worker.start()

    def _run_compare(self) -> None:
        redirector = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector
        try:
            skip_rules = self._build_default_skip_rules()
            print(self.t("log_parsing"), flush=True)
            new_tree, doc_format = tg.parse_strings_xml(Path(self.cmp_new_path.get()))
            old_src_tree, _ = tg.parse_strings_xml(Path(self.cmp_old_source_path.get()))
            old_trans_tree, _ = tg.parse_strings_xml(Path(self.cmp_old_trans_path.get()))

            all_targets = list(tg.iter_translatable_elements(new_tree.getroot(), skip_rules))
            new_targets = [t for t in all_targets if not t.skip]
            old_src_targets = [t for t in tg.iter_translatable_elements(old_src_tree.getroot(), skip_rules)
                               if not t.skip]
            old_trans_targets = [t for t in tg.iter_translatable_elements(old_trans_tree.getroot(), skip_rules)
                                 if not t.skip]

            report = tg.merge_by_locid(new_targets, old_src_targets, old_trans_targets)
            values = tg.seed_list_from_report(report)

            # Pre-fill from the cache (by content) so the table reflects real coverage.
            cache_filled = 0
            cache_path = self.cmp_cache_path.get().strip()
            if cache_path and Path(cache_path).exists():
                try:
                    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
                    for i, entry in enumerate(report.entries):
                        if not values[i]:
                            cached = cache.get(tg.protected_cache_key(entry.new_source))
                            if cached and cached.strip():
                                values[i] = cached
                                cache_filled += 1
                except Exception as exc:
                    print(self.t("log_cache_read_err", exc=exc), flush=True)

            c = report.counts
            print(self.t("log_merge_summary", unchanged=c.get('unchanged', 0),
                         changed=c.get('changed', 0), new=c.get('new', 0),
                         seeded=c.get('seeded', 0), cache=cache_filled), flush=True)

            def _finish() -> None:
                self.cmp_entries = report.entries
                self.cmp_values = values
                self.cmp_edited = set()
                self.cmp_new_targets = new_targets
                self.cmp_all_targets = all_targets
                self.cmp_tree = new_tree
                self.cmp_doc_format = doc_format
                self.cmp_elements = [t.element for t in all_targets]
                self._cmp_populate_tree()
                self.status_label.configure(text=self.t("status_compare_ready"), foreground=FG_SUCCESS)
            self.root.after(0, _finish)
        except Exception as exc:
            print(f"\n❌ Error: {exc}", flush=True)
            self.root.after(0, lambda: self.status_label.configure(
                text=self.t("status_failed"), foreground=FG_ERROR))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    # ------------------------------------------------ compare: table view --

    def _cmp_status_label(self, i: int) -> str:
        entry = self.cmp_entries[i]
        if i in self.cmp_edited:
            return self.t("st_edited")
        if self.cmp_values[i]:
            if entry.reason == "reuse-cosmetic":
                return self.t("st_reuse_cosmetic")
            if entry.reason == "kept-as-source":
                return self.t("st_kept")
            if entry.status == "new":
                return self.t("st_new_translated")
            if entry.status == "changed":
                return self.t("st_changed_translated")
            return self.t("st_reused")
        # pending
        if entry.status == "changed":
            return self.t("st_changed_pending")
        if entry.status == "new":
            return self.t("st_new_pending")
        return self.t("st_pending")

    def _cmp_row_tag(self, i: int) -> str:
        if not self.cmp_values[i]:
            return "pending"
        if i in self.cmp_edited:
            return "ok"
        status = self.cmp_entries[i].status
        if status == "changed":
            return "changed"
        if status == "new":
            return "new"
        return "ok"

    def _cmp_row_matches_filter(self, i: int) -> bool:
        code = self.cmp_filter_code
        entry = self.cmp_entries[i]
        has_value = bool(self.cmp_values[i])
        if code == "diff" and entry.status not in ("changed", "new"):
            return False
        if code == "pending" and has_value:
            return False
        if code == "format" and entry.reason != "reuse-cosmetic":
            return False
        if code == "changed" and entry.status != "changed":
            return False
        if code == "new" and entry.status != "new":
            return False
        if code == "reused" and not has_value:
            return False
        query = self.cmp_search.get().strip().lower()
        if query:
            hay = f"{entry.loc_id or ''} {entry.new_source} {self.cmp_values[i]}".lower()
            if query not in hay:
                return False
        return True

    def _cmp_populate_tree(self) -> None:
        if not hasattr(self, "cmp_treeview"):
            return
        self.cmp_treeview.delete(*self.cmp_treeview.get_children())
        shown = 0
        matched = 0
        for i in range(len(self.cmp_entries)):
            if not self._cmp_row_matches_filter(i):
                continue
            matched += 1
            if shown >= self.CMP_MAX_ROWS:
                continue
            entry = self.cmp_entries[i]
            self.cmp_treeview.insert(
                "", "end", iid=str(i),
                values=(entry.loc_id or "—", entry.new_source, entry.english_old or "—",
                        self.cmp_values[i], self._cmp_status_label(i)),
                tags=(self._cmp_row_tag(i),),
            )
            shown += 1
        total = len(self.cmp_entries)
        pending = sum(1 for v in self.cmp_values if not v)
        note = self.t("cmp_note", matched=matched, shown=shown)
        if matched > shown:
            note += self.t("cmp_note_cap", cap=self.CMP_MAX_ROWS)
        self.cmp_counts_label.configure(text=self.t("cmp_counts", total=total, pending=pending, note=note))

    def _render_diff(self, new_text: str, old_text) -> None:
        """Fill the read-only diff pane with NEW vs OLD English, per-word highlighted."""
        w = self.cmp_diff
        w.configure(state="normal")
        w.delete("1.0", "end")
        if old_text is None:
            # Brand-new string: there is no old English to diff against.
            w.insert("end", self.t("diff_new") + " ", "label")
            w.insert("end", (new_text or "") + "\n", "ins")
            w.configure(state="disabled")
            return
        ops = difflib.SequenceMatcher(None, old_text, new_text).get_opcodes()
        w.insert("end", self.t("diff_new") + " ", "label")
        for op, _i1, _i2, j1, j2 in ops:
            seg = new_text[j1:j2]
            if seg:
                w.insert("end", seg, "ins" if op in ("replace", "insert") else "")
        w.insert("end", "\n")
        w.insert("end", self.t("diff_old") + " ", "label")
        for op, i1, i2, _j1, _j2 in ops:
            seg = old_text[i1:i2]
            if seg:
                w.insert("end", seg, "del" if op in ("replace", "delete") else "")
        w.insert("end", "\n")
        if old_text == new_text:
            w.insert("end", self.t("diff_same"), "label")
        w.configure(state="disabled")

    def _on_cmp_select(self, _event=None) -> None:
        sel = self.cmp_treeview.selection()
        if not sel:
            return
        i = int(sel[0])
        entry = self.cmp_entries[i]
        self._render_diff(entry.new_source, entry.english_old)
        self.cmp_editor.delete("1.0", "end")
        value = self.cmp_values[i]
        loc = entry.loc_id or "—"
        if value:
            self.cmp_editor.insert("1.0", value)
            self.cmp_editor_hint.configure(text=self.t("hint_locid", loc=loc))
        elif entry.draft:
            # Show the old translation as a starting point for a changed/new string.
            self.cmp_editor.insert("1.0", entry.draft)
            self.cmp_editor_hint.configure(text=self.t("hint_draft"))
        else:
            self.cmp_editor_hint.configure(text=self.t("hint_no_translation", loc=loc))

    def _on_cmp_apply_edit(self) -> None:
        sel = self.cmp_treeview.selection()
        if not sel:
            messagebox.showinfo(self.t("mb_no_selection_title"), self.t("mb_no_selection_msg"))
            return
        i = int(sel[0])
        text = self.cmp_editor.get("1.0", "end-1c")
        self.cmp_values[i] = text
        self.cmp_edited.add(i)
        # Update the row in place (or drop it if it no longer matches the filter).
        if self._cmp_row_matches_filter(i):
            self.cmp_treeview.item(
                str(i),
                values=(self.cmp_entries[i].loc_id or "—", self.cmp_entries[i].new_source,
                        self.cmp_entries[i].english_old or "—", text, self._cmp_status_label(i)),
                tags=(self._cmp_row_tag(i),),
            )
        else:
            self.cmp_treeview.delete(str(i))

    # ----------------------------------------- compare: auto-translate --

    def _on_cmp_autotranslate(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(self.t("mb_busy_title"), self.t("mb_op_busy_msg"))
            return
        if not self.cmp_entries:
            messagebox.showinfo(self.t("mb_nothing_title"), self.t("mb_compare_first"))
            return
        pending = sum(1 for v in self.cmp_values if not v)
        if pending == 0:
            messagebox.showinfo(self.t("mb_nothing_translate_title"), self.t("mb_nothing_translate_msg"))
            return
        use_api = bool(self.api_key.get().strip())
        if not use_api and not (self.cmp_cache_path.get().strip()
                                and Path(self.cmp_cache_path.get().strip()).exists()):
            messagebox.showerror(self.t("mb_missing_key_title"), self.t("mb_need_key_or_cache"))
            return
        if use_api and not messagebox.askyesno(
                self.t("mb_confirm_title"), self.t("mb_confirm_autotranslate", pending=pending)):
            return

        self.cancel_event = threading.Event()
        self.cmp_autotranslate_btn.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text=self.t("status_translating"), foreground=FG_ACCENT)
        self.worker = threading.Thread(target=self._run_cmp_autotranslate,
                                       args=(use_api,), daemon=True)
        self.worker.start()

    def _run_cmp_autotranslate(self, use_api: bool) -> None:
        redirector = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector
        try:
            import logging
            for handler in list(logging.root.handlers):
                logging.root.removeHandler(handler)
            logging.basicConfig(
                level=logging.DEBUG if self.verbose.get() else logging.WARNING,
                format="%(levelname)s: %(message)s", stream=redirector)

            cache_path = self.cmp_cache_path.get().strip()
            cache_file = Path(cache_path) if cache_path else None
            inners = [t.text for t in self.cmp_new_targets]

            def progress_cb(done: int, total: int) -> None:
                self.root.after(0, self._set_progress, done, total)

            translated, stats = tg.translate_strings(
                inners,
                api_key=self.api_key.get() or None,
                source_lang=self.source_lang.get().strip() or tg.DEFAULT_SOURCE_LANG,
                target_lang=self.target_lang.get(),
                cache_path=cache_file,
                existing_translations=list(self.cmp_values),
                cache_only=not use_api,
                batch_progress_callback=progress_cb,
                cancel_event=self.cancel_event,
            )
            print("\n" + self.t("log_autotranslate_summary",
                                 cache=stats.cache_used, api=stats.api_translated), flush=True)

            def _finish() -> None:
                # Only fill the gaps: never overwrite a seed, a cache hit, or a manual edit
                # (the button translates the *missing* strings, as its label says).
                self.cmp_values = [
                    old if old else new for old, new in zip(self.cmp_values, translated)
                ]
                self._cmp_populate_tree()
                self.cmp_autotranslate_btn.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status_label.configure(text=self.t("status_autotranslate_ready"), foreground=FG_SUCCESS)
            self.root.after(0, _finish)
        except Exception as exc:
            print(f"\n❌ Error: {exc}", flush=True)

            def _fail() -> None:
                self.cmp_autotranslate_btn.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status_label.configure(text=self.t("status_failed"), foreground=FG_ERROR)
            self.root.after(0, _fail)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    # ------------------------------------------ compare: save / export --

    def _on_cmp_save_cache(self) -> None:
        if not self.cmp_entries:
            messagebox.showinfo(self.t("mb_nothing_save_title"), self.t("mb_compare_first"))
            return
        cache_path = self.cmp_cache_path.get().strip()
        if not cache_path:
            chosen = filedialog.asksaveasfilename(
                title=self.t("mb_save_cache_as"), defaultextension=".json",
                filetypes=[(self.t("ft_json"), "*.json")])
            if not chosen:
                return
            cache_path = chosen
            self.cmp_cache_path.set(cache_path)
        path = Path(cache_path)
        try:
            cache = {}
            if path.exists():
                cache = json.loads(path.read_text(encoding="utf-8"))
            written = 0
            for entry, value in zip(self.cmp_entries, self.cmp_values):
                if value and value.strip() and value != entry.new_source:
                    cache[tg.protected_cache_key(entry.new_source)] = value
                    written += 1
            tg._write_cache_atomic(path, cache)
            messagebox.showinfo(self.t("mb_cache_saved_title"),
                                self.t("mb_cache_saved_msg", n=written, path=path))
        except Exception as exc:
            messagebox.showerror(self.t("mb_error_title"), self.t("mb_cache_save_err", exc=exc))

    def _on_cmp_export(self) -> None:
        if not self.cmp_entries or self.cmp_tree is None:
            messagebox.showinfo(self.t("mb_nothing_export_title"), self.t("mb_compare_first"))
            return
        out = filedialog.asksaveasfilename(
            title=self.t("mb_export_as"), defaultextension=".xml",
            filetypes=[("XML", "*.xml")])
        if not out:
            return
        try:
            # Untranslated entries fall back to the new English source text.
            subset = [
                value if value else target.text
                for value, target in zip(self.cmp_values, self.cmp_new_targets)
            ]
            final_texts = tg.assemble_full_texts(self.cmp_all_targets, subset, enforce_skip_integrity=True)
            tg.write_output_snapshot(self.cmp_tree, self.cmp_elements, final_texts,
                                     Path(out), self.cmp_doc_format)
            self.last_output_path = Path(out)
            self.open_folder_button.configure(state="normal")
            pending = sum(1 for v in self.cmp_values if not v)
            messagebox.showinfo(
                self.t("mb_exported_title"),
                self.t("mb_exported_msg", out=out,
                       done=len(self.cmp_values) - pending, pending=pending))
        except Exception as exc:
            messagebox.showerror(self.t("mb_error_title"), self.t("mb_export_err", exc=exc))

    # ---------------------------------------------------------- file queue --

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title=self.t("title_add_xml"),
            filetypes=[(self.t("ft_xml"), "*.xml"), (self.t("ft_all"), "*.*")],
        )
        for p in paths:
            self._append_input_path(Path(p))
        self._refresh_files_tree()

    def _remove_selected_files(self) -> None:
        selected = self.files_tree.selection()
        if not selected:
            return
        remaining = [p for i, p in enumerate(self.input_paths) if f"I{i}" not in selected]
        self.input_paths = remaining
        self._refresh_files_tree()

    def _clear_files(self) -> None:
        self.input_paths = []
        self._refresh_files_tree()

    def _on_drop(self, event) -> None:
        raw = event.data
        paths: List[str] = []
        pattern = re.compile(r"\{([^}]+)\}|(\S+)")
        for match in pattern.finditer(raw):
            p = match.group(1) or match.group(2)
            if p:
                paths.append(p)
        for p in paths:
            candidate = Path(p)
            if candidate.is_file() and candidate.suffix.lower() == ".xml":
                self._append_input_path(candidate)
        self._refresh_files_tree()

    def _append_input_path(self, path: Path) -> None:
        if path not in self.input_paths:
            self.input_paths.append(path)
            if not self.output_path.get().strip():
                self.output_path.set(str(path.parent))

    def _refresh_files_tree(self) -> None:
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        for i, p in enumerate(self.input_paths):
            self.files_tree.insert("", "end", iid=f"I{i}", values=(str(p),))
        n = len(self.input_paths)
        self.file_count_label.configure(
            text=self.t("files_count_one" if n == 1 else "files_count", n=n))

    # ----------------------------------------------------------- browsers --

    def _browse_output_folder(self) -> None:
        path = filedialog.askdirectory(title=self.t("title_select_output"))
        if path:
            self.output_path.set(path)

    def _browse_cache(self) -> None:
        path = filedialog.askopenfilename(
            title=self.t("title_select_cache"),
            filetypes=[(self.t("ft_cache_json"), "*.cache.json"), (self.t("ft_json"), "*.json"),
                       (self.t("ft_all"), "*.*")],
        )
        if path:
            self.cache_file_path.set(path)

    # ---------------------------------------------------------- utilities --

    def _toggle_secret(self, entry: ttk.Entry) -> None:
        entry.configure(show="" if entry.cget("show") else "•")

    def _refresh_advanced_btn(self) -> None:
        arrow = "▾" if self._advanced_open else "▸"
        self._advanced_btn.configure(text=f"{self.t('btn_advanced')} {arrow}")

    def _toggle_advanced(self) -> None:
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self._advanced_frame.grid()
        else:
            self._advanced_frame.grid_remove()
        self._refresh_advanced_btn()

    def _set_run_buttons(self, running: bool) -> None:
        """Enable/disable the Translate + Cache-only + Stop buttons as a group."""
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.cache_only_btn.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _clear_log(self) -> None:
        self._full_log = ""
        self._set_console_text("")
        self.search_text.set("")

    def _open_output_folder(self) -> None:
        if not self.last_output_path:
            return
        folder = self.last_output_path.parent
        try:
            if sys.platform == "win32":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder)], check=False)
        except Exception as exc:
            messagebox.showerror(self.t("mb_error_title"), self.t("mb_open_folder_err", exc=exc))

    # ------------------------------------------------- API key validation --

    def _test_api_key(self) -> None:
        key = self.api_key.get().strip()
        if not key:
            messagebox.showwarning(self.t("mb_no_key_title"), self.t("mb_no_key_msg"))
            return

        def worker() -> None:
            try:
                client = tg.setup_gemini(key, timeout_seconds=15)
                resp = client.models.generate_content(
                    model=tg.DEFAULT_MODEL,
                    contents="Respond with the single word: OK",
                )
                ok = bool(getattr(resp, "text", None))
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        self.t("mb_apikey_title"),
                        self.t("mb_apikey_ok") if ok else self.t("mb_apikey_notext"),
                    ),
                )
            except Exception as exc:
                msg = str(exc)
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        self.t("mb_apikey_title"), self.t("mb_apikey_fail", msg=msg[:500])
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------ automatic paths --

    def _pair_suffix(self) -> str:
        """Filename suffix that partitions automatic paths per language pair.

        The historic default pair (English → Latin American Spanish) keeps the
        legacy names, so existing outputs/caches keep working.
        """
        src = self.source_lang.get().strip() or tg.DEFAULT_SOURCE_LANG
        tgt = self.target_lang.get().strip() or tg.DEFAULT_TARGET_LANG
        if src == tg.DEFAULT_SOURCE_LANG and tgt == tg.DEFAULT_TARGET_LANG:
            return ""
        return f"_{_lang_slug(src)}-{_lang_slug(tgt)}"

    def _resolve_paths(self, input_path: Path) -> "tuple[Path, Path]":
        """(output, cache) paths for one input file, honoring the manual fields."""
        name = f"{input_path.stem}_translated{self._pair_suffix()}{input_path.suffix}"
        folder = self.output_path.get().strip()
        out_path = (Path(folder) / name) if folder else input_path.with_name(name)
        custom_cache = self.cache_file_path.get().strip()
        cache_path = Path(custom_cache) if custom_cache else out_path.with_suffix(out_path.suffix + ".cache.json")
        return out_path, cache_path

    # -------------------------------------------------------- cost estimate --

    def _estimate_cost(self) -> None:
        if not self.input_paths:
            self.estimate_label.configure(text=self.t("est_no_files"))
            return

        try:
            # Strings already in the cache (explicit, or the automatic per-file
            # one the run would use) cost no API.
            caches: dict = {}

            def _cache_for(path: Path) -> dict:
                _, cache_path = self._resolve_paths(path)
                key = str(cache_path)
                if key not in caches:
                    try:
                        caches[key] = (json.loads(cache_path.read_text(encoding="utf-8"))
                                       if cache_path.exists() else {})
                    except Exception:
                        caches[key] = {}
                return caches[key]

            skip_rules = self._build_default_skip_rules()
            total_strings = 0
            pending_strings = 0
            pending_tokens = 0.0
            pending_bytes = 0
            for path in self.input_paths:
                cache = _cache_for(path)
                tree, _ = tg.parse_strings_xml(path)
                for target in tg.iter_translatable_elements(tree.getroot(), skip_rules):
                    if target.skip:
                        continue
                    total_strings += 1
                    cached = cache.get(tg.protected_cache_key(target.text))
                    if cached and cached.strip():
                        continue  # already translated → no API
                    pending_strings += 1
                    pending_tokens += _approx_tokens(target.text)
                    pending_bytes += len(target.text.encode("utf-8"))

            # Input = the strings plus the rules template resent with every batch;
            # output = the translations, billed at the (much higher) output price.
            batches = -(-pending_bytes // tg.MAX_BUDGET_BYTES) if pending_bytes else 0
            template_tokens = _approx_tokens(tg.DEFAULT_PROMPT_CONFIG.compact_template)
            in_tokens = pending_tokens + template_tokens * batches
            out_tokens = pending_tokens * OUTPUT_LENGTH_FACTOR
            cost_usd = (in_tokens * COST_PER_M_INPUT_TOKENS
                        + out_tokens * COST_PER_M_OUTPUT_TOKENS) / 1_000_000
            tok = f"{int(in_tokens + out_tokens):,}"
            cost = f"{cost_usd:.3f}"
            using_cache = any(caches.values())
            if using_cache:
                self.estimate_label.configure(text=self.t(
                    "est_with_cache", pending=pending_strings,
                    cached=total_strings - pending_strings, tokens=tok, cost=cost))
            else:
                self.estimate_label.configure(text=self.t(
                    "est_result", strings=total_strings, tokens=tok, cost=cost))
        except Exception as exc:
            self.estimate_label.configure(text=self.t("est_error", exc=exc))

    def _build_default_skip_rules(self):
        import argparse
        return tg.build_skip_rules(argparse.Namespace(
            skip_symbol=[], skip_symbol_contains=[], skip_symbol_regex=[],
            skip_text_regex=[], no_path_heuristic=False,
        ))

    # --------------------------------------------------- build cache (no API) --

    def _on_build_cache(self) -> None:
        """Build a cache (English→Spanish) from an English XML + its translated XML. No API."""
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(self.t("mb_busy_title"), self.t("mb_op_busy_msg"))
            return
        xml_types = [(self.t("ft_xml"), "*.xml"), (self.t("ft_all"), "*.*")]
        eng = filedialog.askopenfilename(title=self.t("title_build_pick_eng"), filetypes=xml_types)
        if not eng:
            return
        es = filedialog.askopenfilename(title=self.t("title_build_pick_es"), filetypes=xml_types)
        if not es:
            return
        out = filedialog.asksaveasfilename(
            title=self.t("title_build_save"), defaultextension=".json",
            initialfile=Path(es).stem + ".cache.json",
            filetypes=[(self.t("ft_json"), "*.json")])
        if not out:
            return
        # Two files can't reveal a version mismatch, so confirm the English matches the translation.
        if not messagebox.askyesno(self.t("mb_confirm_title"), self.t("mb_build_confirm")):
            return

        self._clear_log()
        self.cancel_event = threading.Event()
        self._set_run_buttons(running=True)
        self.status_label.configure(text=self.t("status_building"), foreground=FG_ACCENT)
        # This button lives on the Compare tab, but the log/status live on the Translator tab,
        # so surface progress on the Compare tab's own counts label too.
        if hasattr(self, "cmp_counts_label"):
            self.cmp_counts_label.configure(text=self.t("status_building"))
        self.worker = threading.Thread(target=self._run_build_cache, args=(eng, es, out), daemon=True)
        self.worker.start()

    def _run_build_cache(self, eng_path: str, es_path: str, out_path: str) -> None:
        redirector = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector
        try:
            import logging
            for handler in list(logging.root.handlers):
                logging.root.removeHandler(handler)
            logging.basicConfig(
                level=logging.DEBUG if self.verbose.get() else logging.WARNING,
                format="%(levelname)s: %(message)s", stream=redirector)

            skip_rules = self._build_default_skip_rules()
            print(self.t("log_parsing"), flush=True)
            eng_tree, _ = tg.parse_strings_xml(Path(eng_path))
            es_tree, _ = tg.parse_strings_xml(Path(es_path))
            eng_targets = [t for t in tg.iter_translatable_elements(eng_tree.getroot(), skip_rules)
                           if not t.skip]
            es_targets = [t for t in tg.iter_translatable_elements(es_tree.getroot(), skip_rules)
                          if not t.skip]
            # new == old_source == English, so every entry is "unchanged"; the seed is the
            # reusable Spanish (with the same placeholder / not-still-English safety guards).
            report = tg.merge_by_locid(eng_targets, eng_targets, es_targets)

            out = Path(out_path)
            cache = {}
            if out.exists():
                try:
                    cache = json.loads(out.read_text(encoding="utf-8"))
                except Exception:
                    cache = {}
            written = 0
            for entry in report.entries:
                if entry.seed:
                    cache[tg.protected_cache_key(entry.new_source)] = entry.seed
                    written += 1
            tg._write_cache_atomic(out, cache)
            print(self.t("log_build_summary", n=written), flush=True)

            def _finish() -> None:
                self._set_run_buttons(running=False)
                self.status_label.configure(text=self.t("status_build_done"), foreground=FG_SUCCESS)
                if hasattr(self, "cmp_counts_label"):
                    self.cmp_counts_label.configure(text=self.t("log_build_summary", n=written))
                self.last_output_path = out
                self.open_folder_button.configure(state="normal")
                messagebox.showinfo(self.t("mb_build_done_title"),
                                    self.t("mb_build_done_msg", n=written, path=out))
            self.root.after(0, _finish)
        except Exception as exc:
            print(f"\n❌ Error: {exc}", flush=True)

            def _fail() -> None:
                self._set_run_buttons(running=False)
                self.status_label.configure(text=self.t("status_failed"), foreground=FG_ERROR)
                if hasattr(self, "cmp_counts_label"):
                    self.cmp_counts_label.configure(text=self.t("status_failed"))
                messagebox.showerror(self.t("mb_error_title"), self.t("mb_build_err", exc=exc))
            self.root.after(0, _fail)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    # -------------------------------------------------------------- search --

    def _apply_search_filter(self) -> None:
        if not hasattr(self, "console"):
            return
        query = self.search_text.get()
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.insert("end", self._full_log)
        self._tag_log_lines()
        if query:
            start = "1.0"
            while True:
                pos = self.console.search(query, start, nocase=True, stopindex="end")
                if not pos:
                    break
                end = f"{pos}+{len(query)}c"
                self.console.tag_add("match", pos, end)
                start = end
        self.console.see("end")
        self.console.configure(state="disabled")

    def _tag_log_lines(self) -> None:
        for line_num, line in enumerate(self._full_log.splitlines(), start=1):
            lower = line.lower()
            if "error" in lower or "❌" in line or "failed" in lower:
                self.console.tag_add("error", f"{line_num}.0", f"{line_num}.end")
            elif "warning" in lower or "⚠" in line:
                self.console.tag_add("warn", f"{line_num}.0", f"{line_num}.end")
            elif "✅" in line or "completed" in lower:
                self.console.tag_add("success", f"{line_num}.0", f"{line_num}.end")

    # ----------------------------------------------------------- run/stop --

    def _on_run_clicked(self) -> None:
        """'Traducir' button: cache + Gemini for the rest."""
        self._start_translation(cache_only=False)

    def _on_cache_only_clicked(self) -> None:
        """'Solo caché' button: apply the cache only, never call Gemini."""
        self._start_translation(cache_only=True)

    def _start_translation(self, cache_only: bool) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(self.t("mb_busy_title"), self.t("mb_busy_msg"))
            return
        if not self.input_paths:
            messagebox.showerror(self.t("mb_no_files_title"), self.t("mb_no_files_msg"))
            return
        for p in self.input_paths:
            if not p.exists():
                messagebox.showerror(self.t("mb_missing_file_title"), self.t("mb_missing_file_msg", p=p))
                return
        if not cache_only and not self.api_key.get().strip():
            messagebox.showerror(self.t("mb_missing_key_title"), self.t("mb_missing_key_msg"))
            return
        if not self._maybe_warn_source_language():
            return

        self.cache_only.set(cache_only)
        self._clear_log()
        self.cancel_event = threading.Event()
        self._set_run_buttons(running=True)
        self.open_folder_button.configure(state="disabled")
        self.status_label.configure(text=self.t("status_running"), foreground=FG_ACCENT)
        self.progress.configure(value=0, maximum=100)

        self.worker = threading.Thread(target=self._run_batch, daemon=True)
        self.worker.start()

    def _maybe_warn_source_language(self) -> bool:
        """Warn when the first queued file doesn't look like the chosen source language.

        Returns False only when the user cancels the run; detector failures or
        inconclusive guesses never block a translation.
        """
        try:
            skip_rules = self._build_default_skip_rules()
            tree, _ = tg.parse_strings_xml(self.input_paths[0])
            sample = []
            for target in tg.iter_translatable_elements(tree.getroot(), skip_rules):
                if not target.skip:
                    sample.append(target.text)
                    if len(sample) >= 200:
                        break
            detected = _detect_language(sample)
            selected = self.source_lang.get().strip() or tg.DEFAULT_SOURCE_LANG
            if not detected or detected == _lang_slug(selected).split("-")[0]:
                return True
            detected_name = SLUG_TO_NAME.get(detected, detected)
            answer = messagebox.askyesnocancel(
                self.t("mb_lang_mismatch_title"),
                self.t("mb_lang_mismatch_msg", detected=detected_name, selected=selected))
            if answer is None:
                return False
            if answer:
                self.source_lang.set(detected_name)
            return True
        except Exception:
            return True

    def _on_stop_clicked(self) -> None:
        if self.cancel_event:
            self.cancel_event.set()
            self.status_label.configure(text=self.t("status_stopping"), foreground=FG_WARN)
            self.stop_button.configure(state="disabled")

    # --------------------------------------------------------- batch runner --

    def _run_batch(self) -> None:
        redirector = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector

        had_error = False
        had_warnings = False
        try:
            import logging
            for handler in list(logging.root.handlers):
                logging.root.removeHandler(handler)
            logging.basicConfig(
                level=logging.DEBUG if self.verbose.get() else logging.WARNING,
                format="%(levelname)s: %(message)s",
                stream=redirector,
            )

            skip_rules = self._build_default_skip_rules()

            for file_idx, input_path in enumerate(self.input_paths, start=1):
                if self.cancel_event and self.cancel_event.is_set():
                    print("\n⚠️  Batch cancelled by user.", flush=True)
                    had_warnings = True
                    break

                print(f"\n{'=' * 60}", flush=True)
                print(f"📂 File {file_idx}/{len(self.input_paths)}: {input_path.name}", flush=True)
                print("=" * 60, flush=True)

                out_path, cache_file = self._resolve_paths(input_path)

                if cache_file.exists():
                    try:
                        preview = json.loads(cache_file.read_text(encoding="utf-8"))
                        total = len(preview)
                        empty = sum(1 for v in preview.values() if not (v or "").strip())
                        print(f"💾 Cache: {cache_file.name} "
                              f"({total} entries: {total - empty} translated, {empty} empty).", flush=True)
                    except Exception as exc:
                        print(f"⚠️  Cache exists but unreadable: {exc}", flush=True)
                else:
                    print(f"💾 No cache at {cache_file.name} (first run).", flush=True)

                tree, doc_format = tg.parse_strings_xml(input_path)
                targets = list(tg.iter_translatable_elements(tree.getroot(), skip_rules))
                elements = [t.element for t in targets]
                translatable_targets = [t for t in targets if not t.skip]
                translatable_texts = [t.text for t in translatable_targets]

                existing = tg.load_existing_translations(out_path, len(translatable_texts), skip_rules)

                def progress_cb(done: int, total: int) -> None:
                    self.root.after(0, self._set_progress, done, total)

                self.root.after(0, self._set_progress, 0, max(1, len(translatable_texts)))

                translated, stats = tg.translate_strings(
                    translatable_texts,
                    api_key=self.api_key.get() or None,
                    source_lang=self.source_lang.get().strip() or tg.DEFAULT_SOURCE_LANG,
                    target_lang=self.target_lang.get(),
                    cache_path=cache_file,
                    existing_translations=existing,
                    protected_terms=list(tg.DEFAULT_PROTECTED_TERMS),
                    acronym_exclude=list(tg.DEFAULT_ACRONYM_EXCLUDE),
                    cache_only=self.cache_only.get(),
                    retry_empty_cache=self.retry_empty.get(),
                    batch_progress_callback=progress_cb,
                    cancel_event=self.cancel_event,
                )

                final_texts = tg.assemble_full_texts(
                    targets, translated, enforce_skip_integrity=True)
                tg.write_output_snapshot(tree, elements, final_texts, out_path, doc_format)
                self.last_output_path = out_path

                pending = max(0, stats.total_strings - stats.cache_used
                              - stats.api_translated - stats.cache_empty_skipped)
                print(f"\n📊 {input_path.name}", flush=True)
                print(f"  Total: {stats.total_strings} | From cache: {stats.cache_used} | "
                      f"API: {stats.api_translated} | Skipped: {stats.cache_empty_skipped}", flush=True)
                if pending > 0:
                    print(f"  ⚠️  Pending: {pending}", flush=True)
                    had_warnings = True
                if stats.cache_empty_skipped > 0:
                    had_warnings = True
                print(f"  ✅ Written: {out_path}", flush=True)

            if had_error:
                self.log_queue.put("__ERROR__")
            elif had_warnings:
                self.log_queue.put("__DONE_WITH_WARNINGS__")
            else:
                self.log_queue.put("__DONE__")

        except Exception as exc:
            import traceback
            print(f"\n❌ Error: {exc}", flush=True)
            if self.verbose.get():
                traceback.print_exc()
            self.log_queue.put("__ERROR__")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _set_progress(self, done: int, total: int) -> None:
        self.progress.configure(value=done, maximum=max(1, total))
        self.status_label.configure(
            text=self.t("progress_batch", done=done, total=total), foreground=FG_ACCENT)

    # ------------------------------------------------------- console pump --

    def _drain_log_queue(self) -> None:
        try:
            while True:
                chunk = self.log_queue.get_nowait()
                if chunk == "__DONE__":
                    self._on_worker_finished(success=True, warnings=False)
                elif chunk == "__DONE_WITH_WARNINGS__":
                    self._on_worker_finished(success=True, warnings=True)
                elif chunk == "__ERROR__":
                    self._on_worker_finished(success=False, warnings=False)
                else:
                    self._append_console(chunk)
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self._drain_log_queue)

    def _append_console(self, text: str) -> None:
        self._full_log += text
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")
        self._tag_log_lines()

    def _set_console_text(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        if text:
            self.console.insert("end", text)
        self.console.configure(state="disabled")

    def _on_worker_finished(self, success: bool, warnings: bool) -> None:
        self._set_run_buttons(running=False)
        if self.last_output_path:
            self.open_folder_button.configure(state="normal")
        if success and not warnings:
            self.status_label.configure(text=self.t("status_done"), foreground=FG_SUCCESS)
            self.progress.configure(value=self.progress["maximum"])
        elif success and warnings:
            self.status_label.configure(text=self.t("status_done_warnings"), foreground=FG_WARN)
        else:
            self.status_label.configure(text=self.t("status_failed"), foreground=FG_ERROR)

    # ------------------------------------------------------ config persist --

    def _config_path(self) -> Path:
        return Path(__file__).with_name(CONFIG_FILENAME)

    def _load_config(self) -> None:
        path = self._config_path()
        if not path.exists():
            self.root.geometry("900x760")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.root.geometry("900x760")
            return

        lang = data.get("lang", "es")
        self.lang.set(lang if lang in ("es", "en") else "es")
        self.source_lang.set(data.get("source_lang", tg.DEFAULT_SOURCE_LANG))
        self.target_lang.set(data.get("target_lang", tg.DEFAULT_TARGET_LANG))
        self.output_path.set(data.get("output_path", ""))
        self.verbose.set(bool(data.get("verbose", False)))
        self.cache_only.set(bool(data.get("cache_only", False)))
        self.retry_empty.set(bool(data.get("retry_empty", False)))
        # Do NOT persist the API key by default — it is a secret.
        self.cmp_old_source_path.set(data.get("cmp_old_source_path", ""))
        self.cmp_old_trans_path.set(data.get("cmp_old_trans_path", ""))
        self.cmp_cache_path.set(data.get("cmp_cache_path", ""))
        geom = data.get("geometry") or "900x760"
        self.root.geometry(geom)

    def _save_config(self) -> None:
        data = {
            "lang": self.lang.get(),
            "source_lang": self.source_lang.get(),
            "target_lang": self.target_lang.get(),
            "output_path": self.output_path.get(),
            "verbose": self.verbose.get(),
            "cache_only": self.cache_only.get(),
            "retry_empty": self.retry_empty.get(),
            "cmp_old_source_path": self.cmp_old_source_path.get(),
            "cmp_old_trans_path": self.cmp_old_trans_path.get(),
            "cmp_cache_path": self.cmp_cache_path.get(),
            "geometry": self.root.geometry(),
        }
        try:
            self._config_path().write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_config()
        self.root.destroy()


def main() -> None:
    # Any startup crash under pythonw.exe would otherwise be invisible (no console):
    # surface it in a messagebox instead.
    try:
        _run_gui()
    except Exception:
        import traceback
        err = traceback.format_exc()
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Startup error / Error al iniciar",
                f"The GUI failed to start. / La interfaz no pudo iniciarse.\n\n{err}",
            )
        except Exception:
            print(err, file=sys.stderr)
        sys.exit(1)


def _run_gui() -> None:
    # Step 1: tell Windows we render at native DPI (must happen BEFORE creating Tk root).
    _enable_windows_dpi_awareness()

    # Step 2: create root — use TkinterDnD root if drag & drop is available.
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    # Step 3: scale Tk's coordinate system to match real DPI.
    _apply_tk_scaling(root)

    # Step 4: boost default fonts slightly so they stay comfortable after scaling.
    import tkinter.font as tkfont
    for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                      "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                      "TkIconFont", "TkTooltipFont"):
        try:
            f = tkfont.nametofont(font_name)
            # Bump up 1 point above whatever the theme default is.
            current = f.cget("size")
            if current < 0:
                # Negative sizes mean pixels — translate roughly to points and bump.
                f.configure(size=max(10, abs(current) - 2))
            else:
                f.configure(size=max(10, current + 1))
        except Exception:
            pass

    TranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
