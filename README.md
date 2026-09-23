# ThoughtWave-OCD-and-Intrusive-Thought-AI-Assistant
ThoughtWave is a support AI tool for adults that experience distressing Intrusive thoughts and urges to seek repeated reassurance. 

# Current MVP
The current ThoughtWave MVP consist of:
MVP architecture
Component	Implementation	Role
Frontend	Streamlit	Chat interface, session selection, and lightweight debug visibility.
Database	SQLite	Stores sessions, messages, timestamps, detector metadata, and response modes.
Embedding model	`sentence-transformers/all-MiniLM-L6-v2`	Converts user messages into normalized semantic vectors.
Vector store	Python in-memory list	Holds vectors for comparison; rebuilt from SQLite on startup.
Similarity	NumPy dot product on normalized vectors	Measures semantic similarity, equivalent to cosine similarity in this case.
Rule detector	Python regular expressions and heuristic scoring	Looks for certainty wording and combines it with semantic repetition.
LLM adapter	OpenAI-compatible OpenRouter client	Generates a natural-language reply using a configurable model.
Prompt policy	System prompt with injected `RESPONSE_MODE`	Guides the reply according to detector output while keeping generation separate from detection.
