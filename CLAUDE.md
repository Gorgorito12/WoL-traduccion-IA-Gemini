# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A localization tool for the **Age of Empires III: Wars of Liberty** mod. It translates the game's
XML string tables (e.g. `stringtabley.xml`) from English into other languages via Google Gemini
(`gemini-2.5-flash`), with aggressive caching so that re-translating across mod versions costs
almost no API calls.

## Commands

```bat
pip install google-genai tqdm          REM core deps; tkinterdnd2 is optional (drag & drop)

REM CLI translation (see docs/USAGE.md for the full flag list)
python translate_gemini.py "INPUT.xml" "OUTPUT.xml" --api-key "KEY" --cache-file "wol_es.cache.json"

REM Cache-only pass: applies cache, never hits the API (preview / rebuild cache)
python translate_gemini.py "INPUT.xml" "OUTPUT.xml" --cache-file "wol_es.cache.json" --cache-only

REM Run the self-tests (the only tests in the repo) and exit
python translate_gemini.py --self-test-quality-gate
python translate_gemini.py --self-test-merge

REM Launch the GUI. Translator.bat is self-sufficient: it resolves a real Python
REM (rejecting the Microsoft Store alias stub), auto-installs Python 3.13 via winget
REM if missing, auto-installs google-genai/tqdm/tkinterdnd2, and launches with the
REM sibling pythonw.exe. Every failure path prints a message and pauses.
python translate_gui.py
```

There is no build, lint, or external test framework. "Tests" = the eleven `self_test_*` functions
in `translate_gemini.py`: `--self-test-quality-gate` runs eight (quality gate, source casing,
glossary, user glossary, LatAm Spanish, markup integrity, misalignment, repair-from) and
`--self-test-merge` runs three (merge by `_locID`, cache-key parity, build-cache). Both flags work
without the `input output` positionals. The API key falls back to the
`GEMINI_API_KEY` / `GOOGLE_API_KEY` env vars.

## Architecture

Two files, one engine. **`translate_gemini.py` is both the CLI and the library; `translate_gui.py`
imports it as `tg` and calls its functions directly** (`tg.translate_strings`,
`tg.parse_strings_xml`, `tg.iter_translatable_elements`, `tg.build_skip_rules`). The GUI does *not*
shell out to the CLI. **Consequence: changing the signature or behavior of `translate_strings`,
`parse_strings_xml`, or `iter_translatable_elements` will break the GUI** — check `translate_gui.py`
when touching them.

### Translation pipeline (the core flow)

`main()` → `parse_strings_xml` → `iter_translatable_elements` → `translate_strings` → write XML.

1. **Parse & preserve format.** `decode_auto` detects BOM/encoding, `DocumentFormat` captures
   encoding + newline + BOM + xml-declaration so `serialize_tree` can write the file back
   byte-faithfully. A custom `CommentedTreeBuilder` keeps XML comments. This fidelity matters — the
   game is picky about its string-table format.
2. **Select elements.** `iter_translatable_elements` yields `<string>` and `<plurals><item>` nodes.
   `should_skip_element` (driven by `SkipRules`) marks folder/path-like and user-excluded nodes as
   `skip=True`; skipped text is passed through untouched and re-verified by `assemble_full_texts`.
3. **Protect, then tokenize.** Before sending text to the model, two layers of masking are applied
   *in order*: `protect_phrases` replaces protected phrases/acronyms with `__PROTECT_#__`, then
   `protect_tokens` replaces placeholders (`%s`, `%1$s`, `\n`, …) with `__TOK#__`. After
   translation, `restore_all_tokens` reverses both. If a protected token goes missing in the output,
   the code falls back to the original source string rather than emit a corrupted string.
4. **Cache lookup.** Keyed by the *protected* source text → translated text. Misses are batched.
5. **Batch & parallelize.** `yield_batches` packs strings up to `MAX_BUDGET_BYTES`; a
   `ThreadPoolExecutor` (`--max-workers`, default 8) runs `translate_batch_with_retry` →
   `translate_batch_gemini`. The model is asked for a strict JSON array; `reconcile_batch_length`
   repairs length mismatches (pad with source / truncate) so a bad response never desyncs the list.
6. **Quality gates (Spanish-target only, applied per result).** `apply_postprocess_overrides`
   (Home City→Metrópoli, team→equipo, game ages like *National Age*→*Edad Nacional*),
   `enforce_acronym_integrity`, `apply_source_casing`, and
   `has_english_residue`. These are gated by `target_is_spanish()` so other target languages are
   unaffected. The forced-terminology rules (Home City, team, game ages) live in one place,
   `SPANISH_GLOSSARY` (a list of `GlossaryEntry`): each entry carries the `prompt_hint` fed to Gemini
   *and* the deterministic `output_fixes` applied here, so both layers share one source of truth and
   adding a term is one entry. `output_fixes` run only when the entry's `source_trigger` matches the
   English original, and are anchored to whole phrases (e.g. `(Era|Edad) Nacional`→`Edad Nacional`) —
   never a bare `Era`→`Edad`, since `Era` is also the verb *was*. Guarded by `self_test_glossary`.

   `GlossaryEntry` also has an optional **`source_block`**: when it matches the English original the
   whole entry is skipped. It is load-bearing, not a nicety — *"Crates of 500 food and Chests of 500
   coin"* correctly yields both *Cajas* and *Cofres*, `Steel Decks` really is *Cubiertas de Acero*,
   and `Aldeano` is legitimate when the source mentions `Villager` alongside `Settler`.

   **Some entries carry a `prompt_hint` but an EMPTY `output_fixes` on purpose** (`crate`,
   `outpost`, `deck`). Those terms change grammatical gender (`Baraja` f → `Mazo` m), so a regex
   swap leaves a broken article and adjective behind (*"Crear una Mazo nueva"*). Gemini conjugates
   correctly, so the rule is preventive only; strings already wrong are found by `--audit-spanish`
   and re-translated after `--purge-audited`. **Do not "helpfully" add `output_fixes` to them.**

   **`normalize_latam_spanish`** runs at both postprocess points, gated by `target_is_spanish()`.
   Scope is deliberately tiny: a closed list of UI imperatives moved to *tú* (`Presione`→`Presiona`)
   and the missing opening `¡`/`¿`. Peninsular `vosotros` forms are **reported, never rewritten** —
   the real strings carry enclitics (`Atacadnos`, `ponedlos`) and irregulars (`sabed`, `Despertad`)
   that a table cannot conjugate, and the obvious markers are landmines: `sed` is the noun *thirst*,
   `os` appears as a **Portuguese** article in a deliberately Portuguese line, and `id` occurs in
   *"ID de Passport"*. `self_test_latam_spanish` pins every one of those negatives.

   **`markup_integrity_ok`** is the one gate that is NOT language-gated: `<color=...>` markup is
   identical in every target. In this game the coloured word is the *counter keyword* (what a unit
   is strong against), so losing it loses information, not styling. Spanish reorders adjective and
   noun ("Nepalese skirmisher" → "Hostigador nepalí") and the model routinely leaves the pair empty
   or drops it — 61 strings in the shipped translation. It only **detects**: once the sentence is
   reordered a regex cannot know where the word went, so `translate_batch_with_retry` retries with
   `STRICT_MARKUP_RULES` exactly the way the residue gate retries, and `MARKUP_PROMPT_RULE` is sent
   up front (only for batches that actually contain markup) to prevent it. Strings that survive all
   retries are kept — a broken tag still beats an untranslated string — but counted in
   `TranslationStats.markup_rejected` and listed by `--audit-spanish`. Guarded by
   `self_test_markup_integrity`.

   **`has_english_residue` fails INTO the source language**: a flagged string is not written and not
   cached, so the English ships. Its stopword list must therefore stay narrow — `original` and
   `versión` are ordinary Spanish, and `of` inside the game title made 144 correct translations ship
   in English until `"Age of Empires III"` (not just the full mod title) was added to
   `DEFAULT_PROTECTED_TERMS`. `TranslationStats.quality_rejected` / `.batch_failed` now count these
   so they stop being silent.

   **User glossary (pair-agnostic, NOT Spanish-gated).** `glossary.txt` next to the script (or CLI
   `--glossary-file`): `source term = target term` lines, `#` comments. Loaded by
   `load_user_glossary`; same two-layer philosophy: `user_glossary_rules(batch, glossary)` injects
   prompt rules **only into batches that contain the term** (via `translate_batch_with_retry`'s
   `base_extra_rules`, composed with the strict-retry rules), and `apply_user_glossary_fixes`
   deterministically replaces a source term the model left untranslated (whole-word for Latin,
   plain replace for CJK) — it runs at both postprocess points, so it also fixes cache hits. It
   canNOT fix a wrong-but-translated synonym (that's the prompt layer's job). Glossary changes do
   not alter cache keys, so already-cached strings keep their old wording. Threaded through
   `translate_strings(user_glossary=...)`; the GUI auto-loads `glossary.txt` on each run and its
   "Glosario…" button (Avanzado) creates/opens it. Guarded by `self_test_user_glossary`.
7. **Reassemble & write.** `assemble_full_texts` merges translated + skipped text back in original
   order; `write_output_snapshot` does an atomic write. `progress_callback` writes a snapshot
   periodically so partial progress survives a crash/cancel.

### The cache contract (most important invariant)

The cache is a JSON dict `{protected_source_string: translation}` written to
`<output>.cache.json` by default, or to `--cache-file` (the recommended "one global cache per
language" pattern — see `docs/CACHE_WORKFLOW.md`).

- **Empty string `""` is a sentinel meaning "needs translation / previously failed", not a real
  translation.** Failed batches and quality-rejected strings are stored as `""` *deliberately* so a
  re-run retries them instead of caching a bad result. `_prune_empty_cache` strips these before
  writing to disk; `--retry-empty-cache` forces re-translation of them.
- **Never let a failed batch write the original English back into the cache as if translated** — that
  silently poisons future runs. The existing code guards this (`translated_batch = None` path).
- Cache writes are debounced (`CACHE_FLUSH_EVERY_N_BATCHES` / `CACHE_FLUSH_EVERY_SECONDS`) and
  atomic (`_write_cache_atomic`).

### Resumability

If the output file already exists, `load_existing_translations` re-reads it and seeds
`existing_translations`, so an interrupted run continues instead of restarting. The GUI adds a
`cancel_event` (Stop button) that cancels pending futures mid-run.

### Version updates: merge by `_locID`

The real stable key in the WoL XML is the `_locID` attribute on every `<String>` (NOT `symbol`,
which is sparse). `TranslationTarget.loc_id` carries it. `merge_by_locid(new_targets,
old_source_targets, old_trans_targets)` carries an old translation onto a new source version by
matching `_locID` (with a content/position fallback for duplicate or missing ids — there are a
handful of each). It classifies via `_classify_merge_entry` and only **seeds the cache for safe
cases**:
- `reuse-safe` — English unchanged + a real (non-English) old translation + matching `%`-format
  placeholders (`placeholders_compatible`, which is case-insensitive so `%S`≡`%s`).
- `reuse-cosmetic` — the English changed **only** in markup (`<color=...>`), case, or whitespace
  (`_normalize_for_change` ignores those); the old translation is reused as-is. The new markup is
  **not** re-applied (cosmetic). These are surfaced separately (GUI filter "Revisar formato") so
  they can be re-translated if the color formatting matters.
- `kept-as-source` — old translation equals the English **and** there is nothing translatable
  (`_has_translatable_text` false: pure markup/format/filenames); kept as-is.

Everything else is **not** seeded: `changed-needs-review` (real text change — old translation shown
as a draft, never seeded), `old-equals-source` (English left untranslated, but has real words →
flagged to translate), `new-needs-translation`, `placeholder-mismatch`, `no-old-translation`. The
seed feeds the normal engine via `seed_list_from_report` → `existing_translations`. CLI:
`--match-by-locid` + `--prev-source` + `--prev-translation` (+ optional `--report`). Self-tests:
`--self-test-merge`.

**Cache-key parity is load-bearing:** `protected_cache_key()` (and `protect_for_cache` /
`_normalize_protection`) is the single source of truth for the cache key, used both by
`translate_strings` and by external callers (merge seeding, GUI manual edits) so everyone writes
the keys the engine reads. `self_test_cache_key_parity` guards this.

### GUI: two tabs

`translate_gui.py` wraps the engine in a `ttk.Notebook` with two tabs. Its `main()` is a thin
try/except around `_run_gui()` (the real startup) that shows any startup exception in a
`messagebox` — under `pythonw.exe` there is no console, so without this a construction-time crash
would be invisible. **Tab 1 ("Traductor")** is
the original batch translator, unchanged — `_build_ui(parent)` builds it. **Tab 2 ("Comparar /
WinMerge", `_build_compare_ui`)** is the version-update view: pick new-English / old-English /
old-translation (+ optional cache), `merge_by_locid`, show a `ttk.Treeview` (columns `_locID | New
English | Old English | Translation | Status`) colored by status, edit translations by hand,
auto-translate the gaps with Gemini, then save to cache + export the adapted XML. The default filter
is `diff` ("Differences") — only rows where the English changed (status `changed`/`new`), i.e. what
actually needs work. Selecting a row renders a WinMerge-style new-vs-old English diff with per-word
`ins`/`del` highlighting (`_render_diff` via `difflib.SequenceMatcher`); `MergeEntry.english_old`
carries the old text for this. The "smart" carry-over is the content-keyed cache: unchanged English
auto-maps to its old Spanish, only changed/new strings cost work — nothing in the game XML is touched
until export. Both tabs share one background `worker` thread, `cancel_event`, log queue, and the
`is_alive()` busy-guard (one operation at a time). Note: the "Auto-traducir faltantes" finish step
only fills empty values (`old if old else new`) so it never clobbers manual edits or cache hits.

**Translator tab actions.** Two buttons drive translation: **Traducir** (`_on_run_clicked`) =
cache + Gemini, and **Solo caché** (`_on_cache_only_clicked`) = cache only, no API — both call
`_start_translation(cache_only)` (there is no "Cache only" checkbox). `_set_run_buttons(running)`
enables/disables the Traducir/Solo caché/Stop trio together. Rarely-used options (Retry empty cache,
Verbose, and the **Protected words** field) live under an **Avanzado ▾** toggle
(`_toggle_advanced`). Protected words (comma-separated, persisted in the config) become
whole-word case-sensitive regexes via `_custom_protected_regex()` — regex, not plain terms,
because `protect_phrases` exact terms match substrings ("Ram" would mask inside "Framework").
That list is passed to BOTH `translate_strings` call sites and to every GUI
`protected_cache_key` call (compare-tab prefill/save/build-cache and `_estimate_cost`) so cache
keys stay identical across tabs; adding a word changes the key of strings containing it — they
re-translate once. `_estimate_cost` subtracts strings
already in the cache the run would actually use (explicit field or the automatic per-file one, via
the same `_resolve_paths`), so the cost reflects only what would actually hit the API. Its cost
model prices input and output separately (`COST_PER_M_INPUT_TOKENS` / `COST_PER_M_OUTPUT_TOKENS` —
output is ~8× input and dominates), adds the prompt template once per estimated batch, and counts
CJK characters as ~1 token each (`_approx_tokens`).

**Language pairs.** The Settings row is a "Translate from / to" pair of editable comboboxes
(`self.source_lang` / `self.target_lang`, persisted in the config); both flow into
`translate_strings` at the two GUI call sites. Because the cache key is the source text with NO
language dimension, one cache must never serve two pairs — the GUI enforces this through naming:
`_resolve_paths(input)` (single source of truth for `_run_batch` and `_estimate_cost`) appends
`_pair_suffix()` to the automatic output name (`X_translated_zh-en.xml`, slugs from `_lang_slug` /
`LANG_SLUGS`), and the automatic cache derives from that output name, so pairs self-partition.
**Exception: the historic default pair English→Latin American Spanish keeps the legacy names**
(`X_translated.xml`, `<output>.cache.json`) so existing outputs/caches keep working. Manual
Output-folder/Cache-file fields always win. The Spanish-only quality gates are unaffected: they
key off `target_is_spanish()`, not the GUI.

**Source-language sanity check.** `_start_translation` calls `_maybe_warn_source_language()`:
it samples the first queued file and runs `_detect_language()` (module-level heuristic — Unicode
script counts decide ja/zh/ko/ru/ar; stopword votes with a 1.5× margin decide Latin-script
languages, `None` = inconclusive). On a mismatch it shows Yes/No/Cancel (Yes switches the source
combobox to the detected language). Deliberately conservative and failure-proof: inconclusive
guesses and detector exceptions never block a run. Translator tab only — the CLI and the Compare
tab are untouched.

**Build cache (no API).** Building a cache from an already-translated XML lives in ONE engine
function, `build_cache_from_translation(source_targets, translated_targets, ...)`, with three
callers: the CLI flag `--build-cache-from` (via `run_build_cache_cli`), the **Translator tab's**
"Generar…" button next to the *Cache file* field, and the **Compare tab's** "Generar caché (sin
API)…" button — the last two share `_on_build_cache` / `_run_build_cache`, which is now a thin
wrapper. No Gemini. It pairs by `_locID` through `merge_by_locid(src, src, translated)` (every
entry compares "unchanged", so `seed` = the reusable translation) and writes
`cache[protected_cache_key(src)] = value`.

**It deliberately does NOT reuse `merge_by_locid`'s seeding policy wholesale.** That policy serves
version updates, where a translation equal to its source means "still needs translating"
(`old-equals-source`, never seeded). When *building* a cache the same input usually means a proper
noun that legitimately survives translation, so `_identical_translation_is_safe` re-decides those:
keep it if `_has_translatable_text` is false (pure markup), else — for a Spanish target only — keep
it unless `has_english_residue` says it is real untranslated English. For non-Spanish targets there
is no such detector, so it falls back to the conservative markup-only rule. On the real WoL Spanish
table this recovers ~7% of the corpus (≈3.2k proper nouns) that the old seed-only loop dropped and
that would otherwise be re-sent to Gemini. **Do not "fix" this by changing `_classify_merge_entry`**
— that would break the version-update flow and `self_test_merge`. Guarded by
`self_test_build_cache`; returns a `BuildCacheStats` (reused / kept-identical / skipped-English /
skipped-placeholder / unmatched) that both the CLI and the GUI print.

Note: `_normalize_protection` always prepends `DEFAULT_PROTECTED_TERMS`, so `protected_cache_key()`
(used by the builder and the compare tab) yields the same key `translate_strings` consumes —
Translator/CLI keys are unchanged (`protect_phrases` is idempotent). The legacy index-based rebuild
(English-in + matching translated-output, `--cache-only`) still exists but pairs by position and
silently discards everything on a length mismatch.

**Hover tooltips.** Every control in both tabs has a `Tooltip` (a hover `Toplevel`) attached via
`self._tip(widget, key)`, where the text is `lambda: self.t(key)` so it follows the language. Tooltip
text lives under `tip_*` keys in `TR`. When adding a control, attach a tooltip the same way.

**Per-user state.** `.translate_gui_config.json` (window geometry, last-used paths/languages) is
written next to the script on close and is **gitignored** — never commit it; it once shipped the
author's personal output path to every user.

**UI language (ES/EN).** All user-facing text comes from the module-level `TR` dict via `self.t(key,
**fmt)`; the language selector lives in a persistent top bar. Switching language calls
`_build_layout()`, which destroys and recreates the notebook in the new language. State survives
because every persistent var/list is created once in `_init_state()` (not in the build methods) and
the console log + comparison table are restored after the rebuild. The filter dropdown stores a
stable `cmp_filter_code` (not the translated label) so `_cmp_row_matches_filter` is language-agnostic.
When adding UI strings, add a key to BOTH `TR["es"]` and `TR["en"]` and use `self.t(...)` — a plain
literal won't follow the toggle. The engine's own stdout (from `translate_gemini.py`) stays English;
only the GUI's widgets, dialogs, status, and its own log lines are translated.

### Provider coupling

All Gemini-specific code lives in exactly two functions: `setup_gemini` (client creation) and
`translate_batch_gemini` (the API call). `DEFAULT_MODEL = "gemini-2.5-flash"` is hardcoded inside
`translate_batch_gemini` — it is not yet a CLI/GUI parameter. Everything else (caching, batching,
retries, token protection, XML I/O) is provider-agnostic. Adding another backend (e.g. an
OpenAI-compatible provider) means adding a parallel pair of functions and a `--provider` switch,
not touching the pipeline.

`translate_batch_gemini` also pins **`temperature=DEFAULT_TEMPERATURE` (0.2) and
`seed=DEFAULT_SEED`** (CLI `--temperature`/`--seed`). This is not a micro-optimization: batches run
in 8 parallel threads with no shared state, so the prompt's "translate identical strings
identically" is unenforceable and the SDK's default temperature of 1.0 made one term come out two
ways *on the same screen* (`deck` → both "mazo" and "baraja"). `seed` is added in its own
try/except because not every `google-genai` release accepts it.

`translate_batch_gemini` sets `thinking_config=ThinkingConfig(thinking_budget=0)`: 2.5 Flash's
default *dynamic thinking* bills its hidden reasoning tokens at the output price and adds nothing
to mechanical translation — do not remove this without a reason (it silently multiplies cost). The
config is built in a try/except so older `google-genai` SDKs without `ThinkingConfig` still work.

### Auditing (`--audit-spanish`)

`run_audit_spanish_cli` takes the XML **pair** (source as `input`, translation as the flag value),
never a lone cache file — most checks must be triggered by the English, or "Tarjeta de Navidad"
(correct for *Christmas Card*) gets flagged and "Cofre" next to *Chest* gets missed. It reports
glossary terminology, strings that shipped untranslated, peninsular forms and tú/usted balance,
punctuation, and finally `_audit_discover_candidates` — a heuristic sweep for terms in **no**
glossary yet. That last section is noisy by construction (casing, inflection, proper nouns), so it
is printed separately and labelled as candidates; never merge it into the precise section.
`--purge-audited CACHE.json` drops the flagged strings from a cache so the next run redoes them —
the escape hatch for terminology that cannot be fixed by regex.

`_audit_misaligned` catches a different failure: strings whose Spanish is **not a translation of
their English at all**, because a cache was once rebuilt **by position** instead of by `_locID` and
a whole block shifted, putting a building's name in its description's slot. The XML and the cache
hold the same wrong pairing, so **the cache cannot be ground truth here** — it agrees with the
error, and re-running alone re-writes it. Only `--purge-audited` + a re-translation fixes it;
`--build-cache-from` prevents it recurring. The rule is deliberately narrow (one side reads like a
sentence, the other like a label) and **symmetric**: these defects come in swapped pairs, so
catching only one half would fix one slot and leave its partner wrong. A looser rule based on
shared proper nouns was tried and rejected — neighbouring entries share names constantly, so it
produced ten times the hits and most were correct translations. It also checks **shifted numbers
between short labels** (`8 Riflemen` → `6 Fusileros`), which the length rule cannot see and which
are the worst variant: the card promises one number of units and the game shows another.

`--repair-from GOOD_CACHE.json` is the remedy: take the value from an older known-good cache
instead of paying to re-translate. `_structurally_broken` decides what qualifies — the slot holds
another string, a `<color>` pair was lost, or the text is still in the source language. **Wording
problems are excluded on purpose**: there the audited cache is the newer, better translation, and
reaching back to an older one would undo recent terminology work. The donor is refused unless it
keeps the placeholders, has real translatable text (the WoL July cache stores a bare
`__PROTECT_0__` as one "translation") and is not broken itself; the donor file is never written
to. Note `_structurally_broken` reuses the audit's five-real-words threshold for "still in the
source language", or the 4463 proper nouns that survive translation would all qualify.
**Repair before purging, and exempt the repaired keys from the purge**, or the purge deletes what
was just fixed. `--purge-audited` also now skips anything the deterministic post-process already
repairs on export — purging those would discard a good translation and pay for it again.

Note `_FORMAT_SPECIFIER_RE` was widened to cover WoL's own `%1s`/`%2.2f` forms (it only matched
`%s` and `%1$s`), so `placeholders_compatible` — and with it the merge guard and the repair
guard — finally sees them.

**Mod terminology lives in `glossary.txt`, not in code.** Twelve Wars of Liberty terms were each
shipping under several Spanish names (`Allotment` alone had four: *Parcela*, *Reparto*,
*Asignación* — the card, the unit and the upgrade looked like three different things). The English
decides the meaning: an *Allotment* "musters" and "contains units", so it is a block of troops
(*Contingente*), never a plot of land. `Lodge` is the Hunter's Lodge (*Cabaña*), never a Masonic
*Logia*; `Square` is the infantry formation (*Cuadro*), and "Spanish Square" is the *Tercio*, not a
plaza. **Read the full English string before adding a term** — picking by frequency alone is
exactly how `Allotment = Asignación` went wrong. Entries whose canon changes grammatical gender
(`Allotment`, `Square`, `Zapotec`) or whose wrong variant is another unit's name (`Conscript` vs
the `Recruit` unit → never rewrite *Recluta*) are prompt-only, with `output_fixes=()`.

## Docs

`docs/USAGE.md` (all flags), `docs/CACHE_WORKFLOW.md` (version-update strategy), and
`docs/TROUBLESHOOTING.md` (429s, encoding issues) are the authoritative references — prefer updating
them over duplicating flag lists here.
