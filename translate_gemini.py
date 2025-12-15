"""Script FINAL: Gemini 2.5 Flash + MULTIHILO (Paralelismo para máxima velocidad)."""

from __future__ import annotations

import argparse
import logging
import re
import time
import json
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
MAX_CHARS = 4500 

# ¡AQUÍ ESTÁ LA MAGIA! 
# Número de lotes que se traducirán AL MISMO TIEMPO.
# Con tu cuenta de pago, 8 es un número seguro y muy rápido.
MAX_WORKERS = 8 

DEFAULT_MAX_RETRIES = 5
BACKOFF_SECONDS = 1.0

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

def decode_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8")

def yield_batches(strings: Iterable[str], max_chars: int) -> Iterator[List[str]]:
    batch: List[str] = []
    current_len = 0
    for text in strings:
        text_len = len(text) + 10 
        if batch and (current_len + text_len) > max_chars:
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

def translate_batch_gemini(
    model: genai.GenerativeModel, 
    batch: Sequence[str], 
    source_lang: str, 
    target_lang: str
) -> List[str]:
    
    prompt = f"""
    You are a professional video game localizer specializing in history.
    
    Task: Translate the following strings for 'Age of Empires 3: Wars of Liberty' from {source_lang} to **Latin American Spanish**.

    HISTORICAL CONTEXT (1789-1916):
    - Eras: Napoleonic Wars, Industrial Revolution, WWI.
    - Terminology: Accurate military terms (Muskets, Cannons, Ironclads, Trenches).

    LOCALIZATION RULES (LATAM):
    1. **NO 'Vosotros'**: Use 'Ustedes'.
    2. **Vocabulary**: 'Computadora', 'Costo', 'Tomar', 'Video'.
    3. **Tone**: Neutral Spanish.
    
    CRITICAL TECHNICAL RULES:
    1. DO NOT translate tokens like '__TOK0__'.
    2. Return ONLY a valid JSON list of strings.
    3. The output list must have exactly the same number of elements as the input list.

    Input List:
    {json.dumps(batch)}
    """

    try:
        response = model.generate_content(prompt)
        if not response.parts:
             raise ValueError("Respuesta vacía.")

        cleaned_text = clean_json_response(response.text)
        translations = json.loads(cleaned_text)
        
        if len(translations) != len(batch):
            raise ValueError(f"Tamaño incorrecto: Enviados {len(batch)}, Recibidos {len(translations)}")
            
        return translations

    except Exception as e:
        raise e

def translate_batch_with_retry(
    model, batch, source, target, max_retries
) -> List[str]:
    attempt = 0
    while True:
        try:
            return translate_batch_gemini(model, batch, source, target)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                logging.error(f"Fallo crítico en hilo. Saltando lote.")
                return list(batch) 

            wait = BACKOFF_SECONDS 
            time.sleep(wait)

def translate_strings(
    inners: Iterable[str],
    api_key: str,
    source_lang: str,
    target_lang: str,
    max_chars: int = MAX_CHARS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress_callback: Optional[Callable[[Sequence[str]], None]] = None,
) -> List[str]:
    
    setup_gemini(api_key)
    model = genai.GenerativeModel(DEFAULT_MODEL)

    protected: List[str] = []
    token_maps: List[Dict[str, str]] = []
    translations: List[str] = []
    indexes_by_protected: Dict[str, List[int]] = {}
    
    for idx, inner in enumerate(inners):
        protected_text, token_map = protect_tokens(inner)
        protected.append(protected_text)
        token_maps.append(token_map)
        translations.append(inner)
        indexes_by_protected.setdefault(protected_text, []).append(idx)

    cache: Dict[str, str] = {}
    unique_to_translate: List[str] = []
    for text in protected:
        if not text.strip():
            cache[text] = text
            continue
        if text not in cache:
            cache[text] = ""
            unique_to_translate.append(text)

    # Creamos todos los lotes
    batches = list(yield_batches(unique_to_translate, max_chars))
    
    # Mapa para ordenar resultados: {indice_lote: [textos_originales]}
    batch_map = {i: batch for i, batch in enumerate(batches)}
    total_batches = len(batches)

    print(f"🚀 Iniciando motor MULTIHILO: {MAX_WORKERS} trabajadores simultáneos...")

    # --- PROCESAMIENTO PARALELO ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Lanzamos todas las tareas
        future_to_batch_idx = {
            executor.submit(translate_batch_with_retry, model, batch, source_lang, target_lang, max_retries): idx 
            for idx, batch in batch_map.items()
        }

        # Procesamos conforme terminan
        for future in tqdm(as_completed(future_to_batch_idx), total=total_batches, desc="Traduciendo en Paralelo", unit="lote"):
            batch_idx = future_to_batch_idx[future]
            original_batch = batch_map[batch_idx]
            
            try:
                translated_batch = future.result()
            except Exception as exc:
                logging.error(f"Excepción no manejada en hilo: {exc}")
                translated_batch = original_batch # Fallback

            # Guardamos en caché y actualizamos lista principal
            for original, translated_item in zip(original_batch, translated_batch):
                cache[original] = translated_item
                for idx in indexes_by_protected.get(original, []):
                    translations[idx] = unprotect_tokens(translated_item, token_maps[idx])
            
            # Guardamos progreso parcial (thread-safe porque estamos en el hilo principal)
            if progress_callback:
                progress_callback(list(translations))

    return translations

# --- XML Utils ---
def parse_strings_xml(path: Path) -> ET.ElementTree:
    content = decode_auto(path)
    return ET.ElementTree(ET.fromstring(content))

def iter_translatable_elements(root: ET.Element) -> Iterator[ET.Element]:
    def tag_matches(tag: str, name: str) -> bool:
        return tag.split("}")[-1].lower() == name
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

def write_output_snapshot(tree, elements, texts, output):
    update_elements_text(elements, texts)
    tree.write(output, encoding="utf-8", xml_declaration=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target", default=DEFAULT_TARGET_LANG)
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe: {args.input}")

    tree = parse_strings_xml(args.input)
    elements = list(iter_translatable_elements(tree.getroot()))
    
    print(f"🔥 MODO POTENCIA: {DEFAULT_MODEL} + {MAX_WORKERS} Hilos.")

    texts = extract_texts(elements)
    write_output_snapshot(tree, elements, texts, args.output)

    try:
        translated = translate_strings(
            texts,
            api_key=args.api_key,
            source_lang=args.source,
            target_lang=args.target,
            progress_callback=lambda current: write_output_snapshot(
                tree, elements, current, args.output
            )
        )
        write_output_snapshot(tree, elements, translated, args.output)
        print(f"\n✅ Terminado: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()