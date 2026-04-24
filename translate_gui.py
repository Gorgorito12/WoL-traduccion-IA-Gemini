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
        "Missing module",
        f"Could not import translate_gemini.py.\n\n"
        f"Make sure translate_gui.py is in the same folder as translate_gemini.py.\n\n"
        f"Details: {exc}",
    )
    sys.exit(1)


APP_TITLE = "Gemini XML Translator"
CONFIG_FILENAME = ".translate_gui_config.json"

# Rough cost per 1M input tokens for gemini-2.5-flash (USD). Update if pricing changes.
COST_PER_M_INPUT_TOKENS = 0.30
# Approximation: each character ≈ 0.25 tokens.
CHARS_PER_TOKEN = 4

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


class TranslatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(860, 720)
        self.root.configure(bg=BG_MAIN)

        # State ---------------------------------------------------------------
        self.input_paths: List[Path] = []
        self.output_path = tk.StringVar()
        self.api_key = tk.StringVar(
            value=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        )
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

        self._load_config()
        self._apply_dark_theme()
        self._build_ui()

        self.root.after(80, self._drain_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        self.root.option_add("*TCombobox*Listbox*Background", BG_ENTRY)
        self.root.option_add("*TCombobox*Listbox*Foreground", FG_MAIN)
        self.root.option_add("*TCombobox*Listbox*selectBackground", "#264f78")

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- File queue -----------------------------------------------------
        files_frame = ttk.LabelFrame(self.root, text="Input files (queue)", padding=8)
        files_frame.pack(fill="x", padx=10, pady=(10, 4))

        list_container = ttk.Frame(files_frame)
        list_container.pack(fill="x")

        self.files_tree = ttk.Treeview(
            list_container, columns=("path",), show="headings",
            height=4, selectmode="extended",
        )
        self.files_tree.heading("path", text="XML files (drag & drop supported, or use Add…)")
        self.files_tree.column("path", anchor="w")
        self.files_tree.pack(side="left", fill="x", expand=True)

        scroll = ttk.Scrollbar(list_container, orient="vertical", command=self.files_tree.yview)
        scroll.pack(side="right", fill="y")
        self.files_tree.configure(yscrollcommand=scroll.set)

        file_buttons = ttk.Frame(files_frame)
        file_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(file_buttons, text="Add…", command=self._add_files).pack(side="left")
        ttk.Button(file_buttons, text="Remove selected",
                   command=self._remove_selected_files).pack(side="left", padx=4)
        ttk.Button(file_buttons, text="Clear", command=self._clear_files).pack(side="left")

        self.file_count_label = ttk.Label(file_buttons, text="0 files", style="Muted.TLabel")
        self.file_count_label.pack(side="right")

        if DND_AVAILABLE:
            self.files_tree.drop_target_register(DND_FILES)
            self.files_tree.dnd_bind("<<Drop>>", self._on_drop)
        else:
            ttk.Label(
                files_frame,
                text="(Install 'tkinterdnd2' via pip to enable drag & drop.)",
                style="Muted.TLabel",
            ).pack(anchor="w", pady=(4, 0))

        # --- Output + cache -------------------------------------------------
        io_frame = ttk.LabelFrame(self.root, text="Output & cache", padding=8)
        io_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(io_frame, text="Output folder:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(io_frame, textvariable=self.output_path).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(io_frame, text="Browse…",
                   command=self._browse_output_folder).grid(row=0, column=2, **pad)

        ttk.Label(io_frame, text="Cache file:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(io_frame, textvariable=self.cache_file_path).grid(row=1, column=1, sticky="ew", **pad)
        cache_buttons = ttk.Frame(io_frame)
        cache_buttons.grid(row=1, column=2, **pad)
        ttk.Button(cache_buttons, text="Browse…", command=self._browse_cache).pack(side="left")
        ttk.Button(cache_buttons, text="Auto",
                   command=lambda: self.cache_file_path.set("")).pack(side="left", padx=(4, 0))

        ttk.Label(io_frame, text="(Leave fields empty for automatic paths)",
                  style="Muted.TLabel").grid(row=2, column=1, sticky="w", padx=8)
        io_frame.columnconfigure(1, weight=1)

        # --- Settings -------------------------------------------------------
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=8)
        settings_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(settings_frame, text="Target language:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(
            settings_frame, textvariable=self.target_lang,
            values=DEFAULT_TARGET_OPTIONS, width=28,
        ).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(settings_frame, text="Gemini API key:").grid(row=1, column=0, sticky="w", **pad)
        api_entry = ttk.Entry(settings_frame, textvariable=self.api_key, show="•")
        api_entry.grid(row=1, column=1, sticky="ew", **pad)
        btn_row = ttk.Frame(settings_frame)
        btn_row.grid(row=1, column=2, **pad)
        ttk.Button(btn_row, text="Show/Hide",
                   command=lambda: self._toggle_secret(api_entry)).pack(side="left")
        ttk.Button(btn_row, text="Test", command=self._test_api_key).pack(side="left", padx=(4, 0))

        options_row = ttk.Frame(settings_frame)
        options_row.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 2))
        ttk.Checkbutton(options_row, text="Cache only",
                        variable=self.cache_only).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(options_row, text="Retry empty cache",
                        variable=self.retry_empty).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(options_row, text="Verbose", variable=self.verbose).pack(side="left")

        settings_frame.columnconfigure(1, weight=1)

        # --- Action bar -----------------------------------------------------
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=10, pady=(6, 4))

        self.estimate_label = ttk.Label(action_frame, text="", style="Accent.TLabel")
        self.estimate_label.pack(side="left", padx=(0, 10))
        ttk.Button(action_frame, text="Estimate cost",
                   command=self._estimate_cost).pack(side="left")

        self.open_folder_button = ttk.Button(
            action_frame, text="Open output folder",
            command=self._open_output_folder, state="disabled")
        self.open_folder_button.pack(side="right")

        # --- Run / stop / progress ------------------------------------------
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", padx=10, pady=4)

        self.run_button = ttk.Button(run_frame, text="Translate",
                                     style="Accent.TButton", command=self._on_run_clicked)
        self.run_button.pack(side="left")

        self.stop_button = ttk.Button(run_frame, text="Stop",
                                      command=self._on_stop_clicked, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(run_frame, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        self.status_label = ttk.Label(run_frame, text="Idle", style="Muted.TLabel")
        self.status_label.pack(side="right")

        # --- Log console with search ----------------------------------------
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        search_row = ttk.Frame(log_frame)
        search_row.pack(fill="x", pady=(0, 4))
        ttk.Label(search_row, text="Search:").pack(side="left", padx=(0, 4))
        ttk.Entry(search_row, textvariable=self.search_text).pack(
            side="left", fill="x", expand=True)
        self.search_text.trace_add("write", lambda *_: self._apply_search_filter())
        ttk.Button(search_row, text="Clear", command=self._clear_log).pack(side="right")

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

    # ---------------------------------------------------------- file queue --

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add XML files",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
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
        plural = "s" if len(self.input_paths) != 1 else ""
        self.file_count_label.configure(text=f"{len(self.input_paths)} file{plural}")

    # ----------------------------------------------------------- browsers --

    def _browse_output_folder(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_path.set(path)

    def _browse_cache(self) -> None:
        path = filedialog.askopenfilename(
            title="Select cache file",
            filetypes=[("Cache JSON", "*.cache.json"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.cache_file_path.set(path)

    # ---------------------------------------------------------- utilities --

    def _toggle_secret(self, entry: ttk.Entry) -> None:
        entry.configure(show="" if entry.cget("show") else "•")

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
            messagebox.showerror("Error", f"Could not open folder:\n{exc}")

    # ------------------------------------------------- API key validation --

    def _test_api_key(self) -> None:
        key = self.api_key.get().strip()
        if not key:
            messagebox.showwarning("No key", "Please enter an API key first.")
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
                        "API key",
                        "✅ Key valid. Gemini responded successfully."
                        if ok else "⚠️ The call succeeded but returned no text."
                    ),
                )
            except Exception as exc:
                msg = str(exc)
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "API key", f"❌ Key test failed:\n\n{msg[:500]}"
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------- cost estimate --

    def _estimate_cost(self) -> None:
        if not self.input_paths:
            self.estimate_label.configure(text="No files queued.")
            return

        total_strings = 0
        total_chars = 0
        try:
            skip_rules = self._build_default_skip_rules()
            for path in self.input_paths:
                tree, _ = tg.parse_strings_xml(path)
                for target in tg.iter_translatable_elements(tree.getroot(), skip_rules):
                    if not target.skip:
                        total_strings += 1
                        total_chars += len(target.text)
            approx_tokens = total_chars / CHARS_PER_TOKEN
            # Rough upper-bound: ×2.5 accounts for prompt overhead + output tokens.
            cost_usd = (approx_tokens * 2.5) / 1_000_000 * COST_PER_M_INPUT_TOKENS
            self.estimate_label.configure(
                text=f"≈ {total_strings} strings · ~{int(approx_tokens):,} tokens · ~${cost_usd:.3f} USD"
            )
        except Exception as exc:
            self.estimate_label.configure(text=f"Estimate error: {exc}")

    def _build_default_skip_rules(self):
        import argparse
        return tg.build_skip_rules(argparse.Namespace(
            skip_symbol=[], skip_symbol_contains=[], skip_symbol_regex=[],
            skip_text_regex=[], no_path_heuristic=False,
        ))

    # -------------------------------------------------------------- search --

    def _apply_search_filter(self) -> None:
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
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "A translation is already running.")
            return
        if not self.input_paths:
            messagebox.showerror("No files", "Please add at least one XML file.")
            return
        for p in self.input_paths:
            if not p.exists():
                messagebox.showerror("Missing file", f"File does not exist:\n{p}")
                return
        if not self.cache_only.get() and not self.api_key.get().strip():
            messagebox.showerror(
                "Missing API key",
                "Enter a Gemini API key, or enable 'Cache only'.",
            )
            return

        self._clear_log()
        self.cancel_event = threading.Event()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.open_folder_button.configure(state="disabled")
        self.status_label.configure(text="Running…", foreground=FG_ACCENT)
        self.progress.configure(value=0, maximum=100)

        self.worker = threading.Thread(target=self._run_batch, daemon=True)
        self.worker.start()

    def _on_stop_clicked(self) -> None:
        if self.cancel_event:
            self.cancel_event.set()
            self.status_label.configure(text="Stopping…", foreground=FG_WARN)
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

            output_folder = Path(self.output_path.get().strip()) if self.output_path.get().strip() else None
            skip_rules = self._build_default_skip_rules()

            for file_idx, input_path in enumerate(self.input_paths, start=1):
                if self.cancel_event and self.cancel_event.is_set():
                    print("\n⚠️  Batch cancelled by user.", flush=True)
                    had_warnings = True
                    break

                print(f"\n{'=' * 60}", flush=True)
                print(f"📂 File {file_idx}/{len(self.input_paths)}: {input_path.name}", flush=True)
                print("=" * 60, flush=True)

                if output_folder:
                    out_path = output_folder / f"{input_path.stem}_translated{input_path.suffix}"
                else:
                    out_path = input_path.with_name(f"{input_path.stem}_translated{input_path.suffix}")

                custom_cache = self.cache_file_path.get().strip()
                cache_file = Path(custom_cache) if custom_cache else out_path.with_suffix(out_path.suffix + ".cache.json")

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
                    source_lang=tg.DEFAULT_SOURCE_LANG,
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
            text=f"Batch {done}/{total}", foreground=FG_ACCENT)

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
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.last_output_path:
            self.open_folder_button.configure(state="normal")
        if success and not warnings:
            self.status_label.configure(text="Done ✔", foreground=FG_SUCCESS)
            self.progress.configure(value=self.progress["maximum"])
        elif success and warnings:
            self.status_label.configure(text="Done with warnings ⚠", foreground=FG_WARN)
        else:
            self.status_label.configure(text="Failed ✖", foreground=FG_ERROR)

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

        self.target_lang.set(data.get("target_lang", tg.DEFAULT_TARGET_LANG))
        self.output_path.set(data.get("output_path", ""))
        self.verbose.set(bool(data.get("verbose", False)))
        self.cache_only.set(bool(data.get("cache_only", False)))
        self.retry_empty.set(bool(data.get("retry_empty", False)))
        # Do NOT persist the API key by default — it is a secret.
        geom = data.get("geometry") or "900x760"
        self.root.geometry(geom)

    def _save_config(self) -> None:
        data = {
            "target_lang": self.target_lang.get(),
            "output_path": self.output_path.get(),
            "verbose": self.verbose.get(),
            "cache_only": self.cache_only.get(),
            "retry_empty": self.retry_empty.get(),
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
