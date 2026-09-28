# ADR-014: Repo-owned tokenizer read from the GGUF vocabulary

Status: Accepted. Date: 2026-09-28.

## Context

Until this decision Synapse accepted only token IDs. Long-context scenarios were
tokenized by the pinned `llama-tokenize`, stored as ID fixtures, and output IDs
could not be read as text. The owner asked for checks that the model "is not
dumb" on large prompts and for comparisons between systems. Answers, evaluation
text, and chat templates need text in and text out. Without a repo-owned
tokenizer, every evaluation prompt depends on an external tool, and the
generated answers cannot be graded.

GGUF files carry the tokenizer: `tokenizer.ggml.model`, `tokenizer.ggml.pre`,
`tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`, and
`tokenizer.ggml.merges`. The pinned Qwen2.5 file declares `gpt2` byte-level
BPE with the `qwen2` pre-tokenizer.

## Decision

- **Source of truth is the model file.** `TextTokenizers.FromGguf` reads the
  vocabulary, token types, and merge ranks from the GGUF metadata arrays. The
  header reader records each array's element type, count, and offset without
  materializing it. The tokenizer parses strings from the read-only mapping,
  and every length is bounds-checked against the file.
- **One implemented algorithm, explicit refusal otherwise.** The only
  supported algorithm is byte-level BPE (the GPT-2 byte-to-Unicode table plus
  lowest-rank pair merging) with the Qwen2 pre-tokenizer regular expression.
  Any other `tokenizer.ggml.model` or `tokenizer.ggml.pre` value fails with
  `NotSupportedException`. Nothing approximates an unknown tokenizer.
- **Special tokens are explicit.** Control (type 3) and user-defined (type 4)
  tokens are matched whole, longest first, when special parsing is on. Decoding
  omits control tokens unless asked. `ChatTemplates.Qwen` renders ChatML
  as text with explicit special tokens.
- **Public surface.** `ITextTokenizer` (`Encode`, `Decode`, `VocabularySize`),
  `TextTokenizers`, `ChatMessage`, and `ChatTemplates`. The CLI adds
  `synapse tokenize` and `synapse detokenize` over files.
- **Parity is measured, not assumed.** Reference IDs come from the pinned
  llama.cpp `llama-tokenize` over the same model file. The tokenizer is
  qualified only by whole-corpus parity in both special-token modes.
  llama.cpp tools process `\n`-style escapes in their input by default, so
  parity runs pass `--no-escape`.

## Consequences

- Evaluation prompts are written as text and tokenized by Synapse. The same
  text goes to other engines, and each engine's token IDs are checked for
  identity before its answer is compared.
- Families with other tokenizers (SentencePiece, other pre-tokenizers) fail
  explicitly until a separate change implements and qualifies them.
- Tests: `EncodeMatchesLlamaCppReference`, `DecodeRoundTripsText`,
  `SpecialTokensAreParsedAndRenderedOnlyWhenAsked`,
  `ChatTemplateReproducesPasskeyFixture`,
  `UnsupportedPreTokenizerFailsExplicitly`, and the CLI tokenize round trip.
