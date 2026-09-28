# Tokenization

Decision: ADR-014. The tokenizer lives in
`src/Synapse.Runtime/Features/Tokenization`, and the CLI commands live in
`src/Synapse.Cli/Features/Tokenization`.

## Requirements

- `REQ-TOK-001`: Synapse turns text into the model's token IDs and back
  using only the vocabulary, token types, and merges stored in the model file.
  It needs no external tokenizer, no Python, and no network. An unsupported
  tokenizer algorithm or pre-tokenizer fails explicitly.
- `REQ-TOK-002`: Special tokens are handled explicitly. Control tokens are
  parsed whole only when asked and rendered only when asked. The Qwen ChatML
  template is rendered as text with explicit special tokens.
- `REQ-TOK-003`: Parity with the pinned reference tokenizer is measured over
  a whole corpus in both special-token modes. It is not assumed from a few
  strings.

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-TOK-001-1` eleven reference strings (ASCII, Cyrillic, CJK, emoji, code, whitespace runs, contractions) encode to the IDs from `llama-tokenize` b29c606e2 | `TEST-TOK-001-1` `EncodeMatchesLlamaCppReference` |
| `AC-TOK-001-2` decoding returns the original text | `TEST-TOK-001-2` `DecodeRoundTripsText` |
| `AC-TOK-001-3` a tokenizer other than `gpt2` + `qwen2` fails with `NotSupportedException` | `TEST-TOK-001-3` `UnsupportedPreTokenizerFailsExplicitly` |
| `AC-TOK-002-1` `<\|im_start\|>`/`<\|im_end\|>` parse as single IDs only with special parsing on, and render only when asked | `TEST-TOK-002-1` `SpecialTokensAreParsedAndRenderedOnlyWhenAsked` |
| `AC-TOK-002-2` the ChatML template reproduces the locked pass-key fixture IDs | `TEST-TOK-002-2` `ChatTemplateReproducesPasskeyFixture` |
| `AC-TOK-002-3` `synapse tokenize`/`detokenize` round-trip text with special tokens, Cyrillic, and literal escapes; IDs outside the vocabulary fail with exit 1 | `TEST-TOK-002-3` `CliTokenizeAndDetokenizeRoundTrip`, `CliDetokenizeRejectsIdsOutsideTheVocabulary` |
| `AC-TOK-003-1` every tracked text file of the repository tokenizes identically to `llama-tokenize --no-escape` with and without special parsing | `TEST-TOK-003-1` corpus parity run (evidence below) |

## Evidence

**Corpus parity, 2026-09-28** (owner M2 Pro).
- **Scope.** 351 tracked and new text files: `*.md`, `*.cs`, `*.rs`,
  `*.metal`, `*.cu`, `*.cuh`, `*.toml`, `*.yml`, `*.props`, and `*.json`
  under 400 KB.
- **Method.** `synapse tokenize` was compared with
  `llama-tokenize -m qwen2.5-0.5b-instruct-q8_0.gguf -f <file> --ids --no-escape`,
  once with special parsing and once with `--no-parse-special`.
- **Result.** 702 of 702 runs were identical, covering 540,679 tokens in
  literal mode.
- **Pitfall found.** llama.cpp tools process `\n`, `\t`, `\'`, `\"`, and
  `\\` escapes in their input by default. Without `--no-escape`, the first
  file with a literal `\n` diverged: token 198 (newline) against 1699 (`\n`
  as text). That was a reference-tool input transformation, not a tokenizer
  difference.
- **Second pitfall.** llama.cpp `-f` also drops one trailing newline from
  prompt files, while `llama-tokenize` does not. Harnesses that feed
  llama.cpp a prompt ending in `\n` must add one extra newline and check
  the prompt token count that the process itself reports.

**Pinned evaluation corpora** (built from commit `b090e95`, sorted paths,
one blank line between files).
- `corpus-all`: `*.md`, `*.cs`, and `*.rs`, 234 files, 255,478 tokens. Text
  SHA-256 `a3963e41…`, token-ID SHA-256 `a390b5e8…`. The IDs match
  `llama-tokenize` exactly.
- `haystack`: the 38 `*.md` files, then the `*.cs` and `*.rs` files. Text
  SHA-256 `5f19e616…`.
