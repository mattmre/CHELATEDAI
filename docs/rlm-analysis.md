# RLM (Recursive Language Models) Source Code Analysis

Paper: arXiv 2512.24601 by Alex L. Zhang, Tim Kraska, Omar Khattab (MIT)
Repo: https://github.com/alexzhang13/rlm (cloned to `rlm_reference/`)
Analysis date: 2026-02-12

## 1. High-Level Architecture

The RLM system is a **REPL-mediated recursive inference engine**. Rather than decomposing tasks via explicit planning, RLM gives an LLM access to a Python REPL and lets it *programmatically* decide how to decompose, chunk, and recursively call sub-LLMs.

### Core Flow
```
User -> RLM.completion(prompt)
  -> Setup REPL environment with `context` variable
  -> Loop (up to max_iterations):
       -> Send prompt to root LLM (with system prompt + message history)
       -> Parse LLM response for ```repl``` code blocks
       -> Execute each code block in REPL (which has llm_query/llm_query_batched)
       -> Check for FINAL() or FINAL_VAR() in response
       -> If found: return answer
       -> If not: append iteration results to history, continue
  -> If max_iterations exceeded: force a final answer
```

## 2. Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `RLM` | `rlm/core/rlm.py` | Main entry point (~370 lines). Orchestrates completion loop |
| `LMHandler` | `rlm/core/lm_handler.py` | Multi-threaded TCP server routing LLM requests |
| `BaseLM` | `rlm/clients/base_lm.py` | Abstract base for LM clients (completion, usage tracking) |
| `BaseEnv` | `rlm/environments/base_env.py` | Abstract base for REPL environments |
| `LocalREPL` | `rlm/environments/local_repl.py` | Local exec-based REPL with sandboxed namespace |
| `RLMIteration` | `rlm/core/types.py` | Single iteration data (prompt, response, code blocks, final answer) |
| `REPLResult` | `rlm/core/types.py` | Code execution result (stdout, stderr, locals, LLM calls) |
| `RLMChatCompletion` | `rlm/core/types.py` | Completion results (model, prompt, response, usage, time) |
| `LMRequest/LMResponse` | `rlm/core/comms_utils.py` | Typed socket protocol messages |
| `RLMLogger` | `rlm/logger/rlm_logger.py` | JSON-lines logger for trajectory visualization |

## 3. How Recursion Works

**Recursion is NOT explicit tree decomposition.** The LLM writes code that calls `llm_query()`, which triggers a sub-LLM call. The sub-LLM receives whatever prompt the code constructs.

- No explicit task tree -- the LLM decides decomposition via code
- No predefined decomposition strategy -- the LLM writes chunking/aggregation logic
- Sub-calls are just function calls -- `llm_query(prompt)` returns a string

### Depth Management (from `rlm/core/rlm.py`, line 211)
```python
if self.depth >= self.max_depth:
    return self._fallback_answer(prompt)
```

- `depth=0, max_depth=1` (default): Root LLM uses REPL with `llm_query()`, sub-calls are plain LLM calls (no REPL)
- Environment itself gets `depth + 1` passed to `LMRequest.depth`
- `LMHandler.get_client()` routes based on depth: depth=0 uses default, depth=1 uses `other_backend_client`

### Base Case
The LLM signals completion by outputting:
- `FINAL(your answer)` -- inline answer
- `FINAL_VAR(variable_name)` -- retrieve a variable from the REPL

Regex patterns from `rlm/utils/parsing.py`:
```python
final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
final_pattern = r"^\s*FINAL\((.*)\)\s*$"
```

## 4. State Management

### Within a Completion (REPL Variable Persistence)
`LocalREPL` maintains a persistent `self.locals` dict. From `rlm/environments/local_repl.py`:
```python
def execute_code(self, code):
    combined = {**self.globals, **self.locals}
    exec(code, combined, combined)
    for key, value in combined.items():
        if key not in self.globals and not key.startswith("_"):
            self.locals[key] = value
```

Variables from iteration 1 are available in iteration 2, etc.

### Across Completions (Persistent Mode)
When `persistent=True`, the same LocalREPL is reused across multiple `completion()` calls. Contexts accumulate as `context_0`, `context_1`, etc.

### In Isolated Environments
Docker, Modal, Prime, E2B, Daytona environments persist state via `dill` serialization to file.

## 5. Communication Pipeline

### Local (Direct Socket)
```
Code calls llm_query(prompt)
  -> LocalREPL._llm_query() creates LMRequest
  -> TCP socket to LMHandler
  -> Routes to client based on depth
  -> Client calls LLM API
  -> Response flows back through socket
```

### Isolated (HTTP Broker)
```
Code calls llm_query(prompt)
  -> HTTP POST to localhost:8080/enqueue (broker in sandbox)
  -> Broker queues, blocks with threading.Event
  -> Host poller polls /pending via tunnel
  -> Poller forwards to LMHandler
  -> Response posted back to /respond
  -> Broker unblocks, returns to sandbox
```

## 6. Environment Hierarchy

```
BaseEnv (ABC)
  |-- IsolatedEnv (ABC)
  |     |-- ModalREPL, PrimeREPL, E2BREPL, DaytonaREPL
  |-- NonIsolatedEnv (ABC)
        |-- LocalREPL, DockerREPL
```

### LocalREPL Design
- **Safe Builtins**: Blocks `eval`, `exec`, `compile`, `globals`, `locals`, `input`
- **Injected Functions**: `llm_query()`, `llm_query_batched()`, `FINAL_VAR()`, `SHOW_VARS()`
- **Thread Safety**: `threading.Lock` for output capture, UUID temp directory

## 7. Supported Backends (9 total)

| Backend | Client Class | Notes |
|---------|-------------|-------|
| `openai` | `OpenAIClient` | Also for vLLM, OpenRouter, Vercel |
| `anthropic` | `AnthropicClient` | Extracts system messages separately |
| `gemini` | `GeminiClient` | Maps assistant->model role |
| `azure_openai` | `AzureOpenAIClient` | Azure-specific endpoints |
| `portkey` | `PortkeyClient` | Router platform |
| `litellm` | `LiteLLMClient` | Universal router for 100+ providers |
| `vllm` | `OpenAIClient` | Local server via base_url |
| `openrouter` | `OpenAIClient` | Default base_url |
| `vercel` | `OpenAIClient` | Default base_url |

### Model Routing by Depth
From `rlm/core/lm_handler.py`:
```python
def get_client(self, model=None, depth=0):
    if model and model in self.clients:
        return self.clients[model]
    if depth == 1 and self.other_backend_client is not None:
        return self.other_backend_client
    return self.default_client
```

Allows powerful model at root, cheaper model for sub-calls.

## 8. The System Prompt (from `rlm/utils/prompts.py`)

~90 lines of carefully crafted instructions. Key elements:
1. **Context awareness**: Tells LLM about the `context` variable
2. **Sub-LLM tools**: Documents `llm_query`, `llm_query_batched`, `SHOW_VARS`, `print()`
3. **Strategy guidance**: Examples of chunking strategies:
   - Iterative section-by-section with buffer tracking
   - Concurrent batch with `llm_query_batched`
   - Markdown header-based chunking with summaries
4. **Termination**: Clear FINAL()/FINAL_VAR() instructions with common mistake warnings
5. **Action bias**: "Think step by step, plan, and execute immediately"

### User Prompt Construction
- **Iteration 0**: "You have not interacted with the REPL yet" (prevents premature answers)
- **Subsequent**: "The history before is your previous interactions"
- Context metadata injected: "Your context is a str with 50000 total characters, chunks of char lengths: [50000]"

## 9. Parsing and Result Formatting

### Code Block Extraction
```python
pattern = r"```repl\s*\n(.*?)\n```"
```
Only `repl`-tagged blocks are executed. `python`/`bash` blocks are for explanation only.

### Result Formatting
`format_iteration()` creates message history:
- Assistant message with LLM response
- User message per code block: "Code executed: [code]\n\nREPL output: [result]"
- Long outputs truncated to 20000 chars
- Only variable names shown (not values) to save context

## 10. Testing Patterns

| Test File | Focus |
|-----------|-------|
| `tests/test_imports.py` | Module imports, circular dependencies |
| `tests/test_local_repl.py` | REPL basics, persistence, builtins, context |
| `tests/test_local_repl_persistent.py` | Multi-context, multi-history |
| `tests/test_multi_turn_integration.py` | Full RLM with mocked LM, multi-turn |
| `tests/test_parsing.py` | Code block extraction, FINAL/FINAL_VAR parsing |
| `tests/test_types.py` | Data class serialization, QueryMetadata |
| `tests/clients/test_gemini.py` | Gemini client with mocked API |

### Mock Pattern
`MockLM` echoes prompts. Integration tests use `unittest.mock.Mock` with `side_effect`:
```python
mock_lm = Mock()
mock_lm.completion.side_effect = ["response1", "FINAL(answer)"]
```

## 11. Design Patterns

| Pattern | Usage |
|---------|-------|
| Context Manager | `RLM`, environments support `with` for cleanup |
| Factory | `get_client()`, `get_environment()` with lazy imports |
| Protocol (Structural Typing) | `SupportsPersistence` is `@runtime_checkable` Protocol |
| Socket IPC | LMHandler uses 4-byte length prefix + JSON over TCP |
| Broker/Poller | HTTP broker for cross-network communication in isolated envs |
| Iteration-as-Conversation | Each iteration appended as assistant/user messages |

## 12. Comparison: RLM vs ChelatedAI's recursive_decomposer.py

| Aspect | RLM (Official) | ChelatedAI |
|--------|-----------------|------------|
| Decomposition | LLM writes Python code deciding how to decompose | Predefined strategies (MockDecomposer, OllamaDecomposer) |
| Recursion medium | REPL environment -- LLM calls `llm_query()` as function in code | Explicit tree traversal via `_recurse()` |
| State management | Persistent REPL variables (`self.locals` dict) across iterations | DecompositionNode tree with results/scores |
| Base case | LLM signals `FINAL()`/`FINAL_VAR()` when done | `is_base_case()` method on decomposer |
| Depth control | `max_depth` falls back to plain LLM call (no REPL) | `max_depth` stops recursion, does leaf retrieval |
| Aggregation | LLM writes the aggregation code itself | Predefined RRF/union/intersection |
| Sub-call parallelism | `llm_query_batched()` with `asyncio.gather()` | Sequential leaf retrieval |
| Model routing | Depth-based (powerful root, cheap sub-calls) | Single model |
| Self-correction | Iteration feedback loop (sees execution results, retries) | None (single-pass decomposition) |

## 13. Adoptable Patterns for ChelatedAI

### High Priority
1. **Depth-based model routing** -- Use different models for sub-query decomposition vs root queries
2. **Batched sub-calls** -- Add async parallel retrieval for leaf nodes
3. **Iteration-as-conversation** -- Feed decomposition results back for self-correction in OllamaDecomposer
4. **Context metadata** -- Pass character length hints to improve chunking decisions
5. **Forced termination with graceful fallback** -- Already have max_depth; add iteration limits within each node

### Medium Priority
6. **REPL-based decomposer** -- New decomposer class that lets the LLM write decomposition code (RLM's core insight)
7. **Result truncation** -- Cap individual retrieval results to prevent context overflow in aggregation
8. **Trajectory logging** -- Serialize DecompositionTrace in JSONL format for post-training data collection
9. **Variable persistence across queries** -- Maintain a session-level state that accumulates across multiple recursive retrievals

### Lower Priority
10. **Protocol pattern** -- Use `@runtime_checkable` Protocol for duck-typing decomposer capabilities
11. **Socket IPC for parallel retrieval** -- Could enable distributed retrieval across multiple Qdrant instances
12. **Safe builtins list** -- If implementing REPL-based decomposer, adopt the curated builtins approach

## 14. Known Limitations (from CONTRIBUTING.md)

- Only depth 1 currently supported/tested
- No multi-modal input support
- No filesystem/bash environments
- No prefix caching optimization
- No pipelining/async LM call optimization
- Persistent mode only for `local` environment
- No training pipeline for RLM-specific fine-tuning

## 15. Dependencies

From `pyproject.toml`:
- Package: `rlms` on PyPI
- Python: >=3.11
- Core: anthropic, google-genai, openai, portkey-ai, pytest, python-dotenv, requests, rich
- Optional: modal, e2b-code-interpreter, daytona, prime-sandboxes (each with dill)
- Linting: ruff with line-length 100
