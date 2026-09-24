
# ThoughtWave-OCD-and-Intrusive-Thought-AI-Assistant
ThoughtWave is a support AI tool for adults that experience distressing Intrusive thoughts and urges to seek repeated reassurance. 

This solution comes in a form of an AI application that is built as an adaptive, LLM-Integrated adaptive conversational support system. Its core engineering problem is to be able to recognize when a conversation is beginning to function as a reassurance loop and be able to change its response policy before the AI assistant starts to reinforce the repeated certainty seeking.

![ThoughtWave architecture](images/ThoughtWave_Architecture_and_Workflow.png)

The AI System uses two AI model:
- gpt4.0-mini to answer user questions. The ai system uses a system prompt for the llm that can adapt its response based on the behavior state detected by the user, whether an OCD/intrusivethought/ reassurance seeking pattern is detected.
- The chat message history are stored in SQLite DB for the pattern detection
- The huggingface model sentence transformer all-MiniLM-L6-v2 is used to convert the saved message history into vector embeddings stored in memory Vector DB.
- Using cosing similarity between the vector embeddings of the chat history enables the AI system to detect the user's behavior state.


# Run locally
Use Python 3.10 or newer. From the project directory:
```bash
python -m venv .venv
```
Activate the virtual environment:
```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```
Then install dependencies, copy the configuration template, and start Streamlit:
```bash
python -m pip install -r requirements.txt
pip install streamlit   
cp .env.example .env         # Windows PowerShell: Copy-Item .env.example .env
streamlit run app.py
```
Edit `.env` before starting the app:
```dotenv
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL='select-your-model'
HF_TOKEN=
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
`OPENROUTER_API_KEY` is needed for normal chat replies. `HF_TOKEN` for downloading the public embedding model. The first model download can take time. 
Restart Streamlit after changing `.env`.
Do not commit `.env` or API tokens. 
