"""Script FINAL: Gemini 2.5 Flash + MULTIHILO (Paralelismo para máxima velocidad)."""

from __future__ import annotations

import argparse
import logging
import re
import time
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURACIÓN DE POTENCIA ---
DEFAULT_MODEL = "gemini-2.5-flash" 
DEFAULT_SOURCE_LANG = "English"
DEFAULT_TARGET_LANG = "Latin American Spanish"

# Mantenemos lotes medianos para agilidad
MAX_BUDGET_BYTES = 4500

# ¡AQUÍ ESTÁ LA MAGIA! 
# Número de lotes que se traducirán AL MISMO TIEMPO.
# Con tu cuenta de pago, 8 es un número seguro y muy rápido.
DEFAULT_MAX_RETRIES = 5
BACKOFF_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0

# Usa el prompt compacto por defecto para reducir tokens sin perder reglas clave.
DEFAULT_COMPACT_PROMPT = True
DEFAULT_MAX_WORKERS = 8

PLACEHOLDER_RE = re.compile(r"(%\d+\$[sdif]|%[sdif]|\\n|\\t|\\r)")

def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)

def protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}
    idx = 0
    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        key = f"__TOK{idx}__" 
        token_map[key] = match.group(0)
        idx += 1
        return key
    return PLACEHOLDER_RE.sub(repl, text), token_map

def unprotect_tokens(text: str, token_map: Dict[str, str]) -> str:
    for key, value in token_map.items():
        text = text.replace(key, value)
    return text

def decode_auto(path: Path) -> Tuple[str, str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16"), "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16-be"
    return raw.decode("utf-8"), "utf-8"

def yield_batches(strings: Iterable[str], max_budget_bytes: int, max_items: int = 50) -> Iterator[List[str]]:
    batch: List[str] = []
    current_len = 0
    for text in strings:
        text_len = len(text.encode("utf-8")) + 32  # amortiza comillas y tokens
        if batch and (current_len + text_len > max_budget_bytes or len(batch) >= max_items):
            yield batch
            batch = []
            current_len = 0
        batch.append(text)
        current_len += text_len
    if batch:
        yield batch

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def build_prompt(batch: Sequence[str], source_lang: str, target_lang: str, compact: bool) -> str:
    if compact:
        return (
            "Translate for 'Age of Empires III: Wars of Liberty'. "
            f"{source_lang} -> {target_lang}. "
            "Rules: Latin American Spanish, use 'Ustedes'; avoid anachronisms; keep UI tone concise. "
            "Preferred words: computadora, costo (UI), tomar, video. "
            "Respect placeholders (__TOK#, %s, %1$s, %d, \\n, \\t); do not move, delete, or edit them. "
            "Return ONLY a JSON array, same length/order as input; empty/placeholder-only strings stay the same. "
            "If unsure, return the original string. Input list: "
            f"{json.dumps(batch, ensure_ascii=False)}"
        )

    return f"""
    You are a professional video game localization specialist for “Age of Empires III: Wars of Liberty”.

    TASK
    Translate from {source_lang} to {target_lang} with natural, concise UI/game text.

    ERA & STYLE (1789–1916)
    - Use accurate terms for Napoleonic Wars, Industrial Revolution, and WWI.
    - Avoid anachronisms; do not use archaic Spanish.
    - Latin American Spanish; always “Ustedes”, never “Vosotros”.
    - Preferred: computadora, costo (UI), tomar, video. Imperatives: “Presione”, “Seleccione”.

    TECHNICAL RULES (STRICT)
    1. DO NOT translate or alter placeholders (__TOK#, %s, %1$s, %d, \n, \t).
    2. Do not merge/split/rephrase strings.
    3. Output ONLY a JSON array of strings, same length/order as input.
    4. If a string is empty or only placeholders, return it unchanged.
    5. If any rule fails, return the original string unchanged.

    Input List: {json.dumps(batch, ensure_ascii=False)}
    """


def translate_batch_gemini(
    model: genai.GenerativeModel,
    batch: Sequence[str],
    source_lang: str,
    target_lang: str,
    compact_prompt: bool,
) -> List[str]:

    prompt = build_prompt(batch, source_lang, target_lang, compact_prompt)

    response = model.generate_content(prompt)

    if not response.candidates:
        raise ValueError("Respuesta sin candidatos.")

    first_candidate = response.candidates[0]
    finish_reason = getattr(first_candidate, "finish_reason", None)
    normalized_finish = str(finish_reason).lower() if finish_reason else ""

    if not response.parts or not response.text:
        raise ValueError("Respuesta vacía o sin texto utilizable.")

    if finish_reason and normalized_finish != "stop":
        logging.warning(
            "finish_reason inesperado (%s) pero se recibió texto; continuando.",
            finish_reason,
        )

    cleaned_text = clean_json_response(response.text)
    try:
        translations = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido: {exc}. Texto recibido: {cleaned_text[:120]}")

    if len(translations) != len(batch):
        err = ValueError(
            f"Tamaño incorrecto: Enviados {len(batch)}, Recibidos {len(translations)}"
        )
        setattr(err, "partial_translations", translations)
        raise err

    return translations


def is_retryable_error(exc: Exception) -> bool:
    transient_signals = (
        "rate limit",
        "temporarily unavailable",
        "try again",
        "deadline exceeded",
        "overloaded",
    )
    value_error_retryables = (
        "respuesta no completada",
        "respuesta vacía",
        "json inválido",
        "tamaño incorrecto",
        "sin candidatos",
        "finish_reason=safety",
        "finish_reason=blocked",
    )

    message = str(exc).lower()

    if any(signal in message for signal in transient_signals):
        return True

    finish_reason_hints = ("finish_reason=safety", "finish_reason=blocked", "safety", "blocked")

    if any(hint in message for hint in finish_reason_hints):
        return True

    if isinstance(exc, ValueError) and any(signal in message for signal in value_error_retryables):
        return True

    return not isinstance(exc, ValueError)

def translate_batch_with_retry(
    model,
    batch,
    source,
    target,
    max_retries,
    compact_prompt: bool,
) -> List[str]:
    attempt = 0
    last_partial: Optional[List[str]] = None
    while True:
        try:
            return translate_batch_gemini(model, batch, source, target, compact_prompt)
        except Exception as exc:
            attempt += 1
            partial = getattr(exc, "partial_translations", None)
            if partial:
                last_partial = partial
            retryable = is_retryable_error(exc)
            logging.warning(
                "Error en lote (intento %s/%s, reintentar=%s): %s",
                attempt,
                max_retries,
                retryable,
                exc,
            )
            if (not retryable) or attempt > max_retries:
                logging.error("Fallo crítico en hilo. Saltando lote." )
                if last_partial:
                    return last_partial
                return list(batch)

            backoff = min(BACKOFF_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            backoff += random.uniform(0, BACKOFF_SECONDS)
            time.sleep(backoff)

def translate_strings(
    inners: Iterable[str],
    api_key: str,
    source_lang: str,
    target_lang: str,
    max_budget_bytes: int = MAX_BUDGET_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_workers: int = DEFAULT_MAX_WORKERS,
    compact_prompt: bool = DEFAULT_COMPACT_PROMPT,
    progress_callback: Optional[Callable[[Sequence[str]], None]] = None,
    cache_path: Optional[Path] = None,
    existing_translations: Optional[Sequence[str]] = None,
) -> List[str]:
    
    setup_gemini(api_key)
    model = genai.GenerativeModel(DEFAULT_MODEL)

    protected: List[str] = []
    token_maps: List[Dict[str, str]] = []
    translations: List[str] = []
    indexes_by_protected: Dict[str, List[int]] = {}

    cache: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("No se pudo cargar caché previa (%s): %s", cache_path, exc)
            cache = {}

    for idx, inner in enumerate(inners):
        protected_text, token_map = protect_tokens(inner)
        protected.append(protected_text)
        token_maps.append(token_map)

        initial_translation = inner
        if existing_translations and idx < len(existing_translations):
            candidate = existing_translations[idx]
            if candidate and candidate.strip():
                initial_translation = candidate
                if candidate != inner:
                    cache.setdefault(protected_text, candidate)

        translations.append(initial_translation)
        indexes_by_protected.setdefault(protected_text, []).append(idx)
    unique_to_translate: List[str] = []
    already_enqueued: set[str] = set()

    for text in protected:
        if not text.strip():
            cache[text] = text
            # Propagamos el texto vacío tal cual a todas sus posiciones.
            for idx in indexes_by_protected.get(text, []):
                translations[idx] = unprotect_tokens(text, token_maps[idx])
            continue

        cached_value = cache.get(text)

        if cached_value and cached_value.strip():
            # Ya teníamos una traducción previa en caché: la aplicamos a todas
            # las apariciones de este texto y evitamos re-traducirlo.
            for idx in indexes_by_protected.get(text, []):
                translations[idx] = unprotect_tokens(cached_value, token_maps[idx])
            continue

        # Si no hay caché (o está vacía), registramos entrada y lo programamos
        # para traducción, cuidando de no encolar duplicados.
        if text not in cache:
            cache[text] = ""
        if text not in already_enqueued:
            already_enqueued.add(text)
            unique_to_translate.append(text)

    # Creamos todos los lotes
    batches = list(yield_batches(unique_to_translate, max_budget_bytes))
    
    # Mapa para ordenar resultados: {indice_lote: [textos_originales]}
    batch_map = {i: batch for i, batch in enumerate(batches)}
    total_batches = len(batches)

    print(f"🚀 Iniciando motor MULTIHILO: {max_workers} trabajadores simultáneos...")

    # --- PROCESAMIENTO PARALELO ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Lanzamos todas las tareas
        future_to_batch_idx = {
            executor.submit(
                translate_batch_with_retry,
                model,
                batch,
                source_lang,
                target_lang,
                max_retries,
                compact_prompt,
            ): idx
            for idx, batch in batch_map.items()
        }

        # Procesamos conforme terminan
        for future in tqdm(as_completed(future_to_batch_idx), total=total_batches, desc="Traduciendo en Paralelo", unit="lote"):
            batch_idx = future_to_batch_idx[future]
            original_batch = batch_map[batch_idx]
            
            try:
                translated_batch = future.result()
            except Exception as exc:
                logging.error(
                    "Excepción no manejada en hilo (lote %s, %s items): %s",
                    batch_idx,
                    len(original_batch),
                    exc,
                )
                translated_batch = original_batch  # Fallback

            # Guardamos en caché y actualizamos lista principal
            for original, translated_item in zip(original_batch, translated_batch):
                cache[original] = translated_item
                for idx in indexes_by_protected.get(original, []):
                    translations[idx] = unprotect_tokens(translated_item, token_maps[idx])
            
            # Guardamos progreso parcial (thread-safe porque estamos en el hilo principal)
            if cache_path:
                try:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    logging.warning("No se pudo persistir la caché en lote %s: %s", batch_idx, exc)

            if progress_callback:
                progress_callback(list(translations))

    return translations

# --- XML Utils ---
class CommentedTreeBuilder(ET.TreeBuilder):
    """TreeBuilder que conserva comentarios XML al parsear."""

    def comment(self, data):
        self.start(ET.Comment, {})
        self.data(data)
        self.end(ET.Comment)


def parse_strings_xml(path: Path) -> Tuple[ET.ElementTree, str]:
    content, encoding = decode_auto(path)
    parser = ET.XMLParser(target=CommentedTreeBuilder())
    return ET.ElementTree(ET.fromstring(content, parser=parser)), encoding

def iter_translatable_elements(root: ET.Element) -> Iterator[ET.Element]:
    def tag_matches(tag: str, name: str) -> bool:
        if not isinstance(tag, str):
            return False
        # Algunos nodos especiales (p. ej. comentarios) pueden filtrarse con
        # un ``tag`` inesperado; usamos ``split`` de forma segura para evitar
        # AttributeError cuando el tag no es un string normal.
        splitter = getattr(tag, "split", None)
        if splitter is None:
            return False
        return splitter("}")[-1].lower() == name
    for elem in root.iter():
        if tag_matches(elem.tag, "string"):
            yield elem
        elif tag_matches(elem.tag, "plurals"):
            for item in elem:
                if tag_matches(item.tag, "item"):
                    yield item

def extract_texts(elements: Iterable[ET.Element]) -> List[str]:
    return [(elem.text or "") for elem in elements]

def update_elements_text(elements: Iterable[ET.Element], texts: Sequence[str]) -> None:
    for elem, text in zip(elements, texts):
        elem.text = text

def write_output_snapshot(tree, elements, texts, output, encoding: str):
    update_elements_text(elements, texts)
    tree.write(output, encoding=encoding, xml_declaration=True)


def load_existing_translations(path: Path, reference_count: int) -> Optional[List[str]]:
    if not path.exists():
        return None

    try:
        existing_tree, _ = parse_strings_xml(path)
        existing_elements = list(iter_translatable_elements(existing_tree.getroot()))
        if len(existing_elements) != reference_count:
            logging.warning(
                "El archivo de salida existente (%s) no coincide en longitud (esperado %s, encontrado %s). Ignorando.",
                path,
                reference_count,
                len(existing_elements),
            )
            return None
        return extract_texts(existing_elements)
    except Exception as exc:
        logging.warning("No se pudo cargar traducciones previas desde %s: %s", path, exc)
        return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-budget-bytes", type=int, default=MAX_BUDGET_BYTES)
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        dest="compact_prompt",
        default=DEFAULT_COMPACT_PROMPT,
        help="Usa un prompt condensado (por defecto) para gastar menos tokens.",
    )
    parser.add_argument(
        "--detailed-prompt",
        action="store_false",
        dest="compact_prompt",
        help="Usa el prompt detallado si quieres máxima explicitud a costa de más tokens.",
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe: {args.input}")

    tree, input_encoding = parse_strings_xml(args.input)
    elements = list(iter_translatable_elements(tree.getroot()))

    print(f"🔥 MODO POTENCIA: {DEFAULT_MODEL} + {args.max_workers} Hilos.")
    if args.compact_prompt:
        print("💾 Prompt compacto activado (optimiza tokens sin perder reglas).")
    else:
        print("🧭 Prompt detallado activado (más contexto, mayor costo de tokens).")

    texts = extract_texts(elements)
    existing_translations = load_existing_translations(args.output, len(elements))
    starting_texts = existing_translations or texts

    if existing_translations:
        print("↩️  Reanudando traducción desde archivo existente.")

    write_output_snapshot(tree, elements, starting_texts, args.output, input_encoding)

    cache_file = args.output.with_suffix(args.output.suffix + ".cache.json")

    try:
        translated = translate_strings(
            texts,
            api_key=args.api_key,
            source_lang=args.source,
            target_lang=args.target,
            cache_path=cache_file,
            existing_translations=existing_translations,
            max_workers=args.max_workers,
            max_budget_bytes=args.max_budget_bytes,
            compact_prompt=args.compact_prompt,
            progress_callback=lambda current: write_output_snapshot(
                tree, elements, current, args.output, input_encoding
            )
        )
        write_output_snapshot(tree, elements, translated, args.output, input_encoding)
        print(f"\n✅ Terminado: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
