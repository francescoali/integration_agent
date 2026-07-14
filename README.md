# Math Integration Agent

A FastAPI service that wraps an LLM agent capable of translating plain-language, real-world word problems (distance, area, work, accumulated change, etc.) into definite integrals, and evaluating them numerically. The agent runs on a self-hosted [Ollama](https://ollama.com/) model (`llama3.2`, 3B) via [pydantic-ai](https://ai.pydantic.dev/), and delegates all actual math to a SymPy + SciPy tool rather than doing arithmetic itself.

## How it works

1. A user sends a natural-language problem description (e.g. *"A car accelerates at 2 m/s² starting from rest — how far does it travel in the first 5 seconds?"*) to the `/prompt` endpoint.
2. The LLM agent:
   - Identifies the physical/mathematical quantity being requested.
   - Derives the correct integrand as a function of `x` and the correct bounds.
   - Calls the `integrate_expression_tool` function-calling tool to compute the definite integral — the model is instructed to **never** compute the integral or arithmetic itself.
   - Returns the numeric result along with an explanation of the integral it constructed.
3. The tool (`integrate_expression`) parses the expression with **SymPy**, converts it to a fast numeric function with `lambdify`, and integrates it numerically with **SciPy's** `scipy.integrate.quad`.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally, with the `llama3.2` model pulled:
  ```bash
  ollama pull llama3.2
  ```
- Python packages:
  ```bash
  pip install sympy scipy pydantic-ai fastapi uvicorn python-dotenv
  ```

## Configuration

The app loads environment variables from a `.env` file at startup. Create one in the project root with the Ollama server URL, for example:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

## Running the server

Make sure Ollama is running, then start the API:

```bash
python app.py
```

This launches a Uvicorn server at `http://127.0.0.1:8000` with auto-reload enabled.

> **Note:** the script must be saved as `app.py` for the `uvicorn.run("app:app", ...)` reference to resolve correctly.

## API Usage

### `POST /prompt`

Send a plain-language problem as a query parameter (or request body, depending on your FastAPI client) named `prompt`.

**Example (curl):**

```bash
curl -X POST "http://127.0.0.1:8000/prompt?prompt=A%20car%20accelerates%20at%202%20m%2Fs%C2%B2%20from%20rest.%20How%20far%20does%20it%20travel%20in%205%20seconds%3F"
```

**Response:** the full agent run result, including the final answer text and the reasoning/tool-call trace produced by pydantic-ai.

### Interactive docs

FastAPI automatically generates interactive Swagger UI docs. With the server running, open:

```
http://127.0.0.1:8000/docs
```

to try the endpoint directly from your browser.

## Core components

| Component | Purpose |
|---|---|
| `integrate_expression()` | Parses a math expression string with SymPy and numerically integrates it over `[lower_bound, upper_bound]` using `scipy.integrate.quad`. |
| `agent` (pydantic-ai `Agent`) | An LLM agent backed by a local Ollama model, prompted to reason about word problems and translate them into integrals. |
| `integrate_expression_tool` | The agent-callable tool wrapping `integrate_expression`, registered via `@agent.tool_plain`. |
| `/prompt` endpoint | FastAPI route that accepts a user prompt and returns the agent's response. |

## Security notes

- `sp.sympify` is called with a restricted `locals` namespace (`{"x": x}`) to limit what symbols/names are resolvable, but `sympify` can still evaluate arbitrary Python-like expressions — avoid exposing this endpoint to untrusted public traffic without additional input sanitization or sandboxing.
- Consider adding request validation (e.g. a Pydantic model for the request body instead of a raw query parameter) and rate limiting before deploying beyond local/dev use.

## Known limitations

- The agent depends entirely on the LLM correctly identifying the integrand and bounds from natural language; incorrect physical reasoning by the model will produce a confidently wrong (but numerically "correct" for the wrong integral) answer.
- `llama3.2:3b` is a small model — for more reliable integral construction on complex word problems, consider a larger Ollama model.
- No persistence/session memory between requests; each `/prompt` call is a fresh agent run.


## Contact

[LinkedIn](https://www.linkedin.com/in/francesco-al%C3%AC/)
[E-mail](mailto:francescoali@edgine.it)
