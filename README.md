# ThoughtWave MVP

ThoughtWave is a Streamlit portfolio prototype that answers ordinary questions
normally and changes its response policy when repeated certainty seeking becomes
more likely across a conversation.

It detects an observable interaction pattern. It does not diagnose OCD, provide
therapy, or replace emergency support.

## How the two models work together

1. Hugging Face downloads `sentence-transformers/all-MiniLM-L6-v2` by using
   `HF_TOKEN`.
2. The embedding model runs locally and compares the current message with earlier
   user messages in the active conversation.
3. A deterministic detector selects `NORMAL`, `MONITOR`, `POSSIBLE_LOOP`, or
   `STRONG_LOOP`.
4. ThoughtWave builds a state-specific system prompt and sends it, the current
   question, recent history, and relevant earlier messages to OpenRouter.
5. The assistant response appears only after OpenRouter returns a valid, non-empty
   completion.
6. The successful turn is stored in SQLite. The current user message enters
   semantic memory only after comparison, so it cannot match itself.

OpenRouter supplies the conversational answer. The embedding model never generates
an answer, and the application has no canned answer fallback.

## Setup

Python 3.11 or 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and provide:

```env
HF_TOKEN=hf_your_token
OPENROUTER_API_KEY=sk-or-v1-your_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

The selected OpenRouter model must be available to your account. You may replace
the example slug with another supported chat model.

## Run

```bash
streamlit run app.py
```

The first startup can take longer because Hugging Face downloads the embedding
model. Later startups use the local model cache.

## Test

```bash
pytest -q
```

The tests use local fakes and mocked HTTP responses. They do not consume Hugging
Face or OpenRouter credits.

## Four behavioral states

- `NORMAL`: answer the actual question naturally.
- `MONITOR`: remain helpful while avoiding unnecessary absolute guarantees.
- `POSSIBLE_LOOP`: answer concisely and gently note that the conversation may be
  returning to the same uncertainty.
- `STRONG_LOOP`: stop escalating reassurance, tentatively name the repeated
  pattern, and encourage allowing uncertainty while returning to a present task or
  chosen value.

The default engineering score is:

- certainty-seeking wording: `+1`
- highest similarity above threshold: `+2`
- at least two related earlier messages: `+2`
- at least four related earlier messages: `+1`
- escalation wording alongside semantic repetition: `+1`

State ranges are `0–1 NORMAL`, `2–3 MONITOR`, `4–5 POSSIBLE_LOOP`, and `6+ STRONG_LOOP`.
These are demonstration thresholds, not clinical cutoffs.

## Data and privacy

- Chat sessions and messages are stored in `data/thoughtwave.db`.
- Embeddings remain in memory and are rebuilt from prior user messages after restart.
- Semantic search is scoped to the active session.
- API keys are read from `.env` and are never stored in SQLite.
- Do not commit `.env` or the database.

Before any real-user pilot, the project needs clinician review, consent and deletion
controls, encryption, access control, locale-specific safety pathways, and a more
complete independent safety layer.

## Troubleshooting

### The app is stuck while starting

The embedding model may still be downloading. Confirm that `HF_TOKEN` is valid and
that Hugging Face is reachable. The app reports initialization failures instead of
continuing with an unavailable model.

### OpenRouter produces no answer

ThoughtWave reports invalid keys, inaccessible model slugs, insufficient credits,
rate limits, timeouts, malformed responses, and empty responses separately. It does
not replace a failed generation with a fixed therapeutic message.

### `torchvision` is missing

This text-only application does not directly use `torchvision`. Install the supplied
requirements in a clean virtual environment. Do not add `torchvision` merely to run
ThoughtWave unless another package in your own environment explicitly requires it.

