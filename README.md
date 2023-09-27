## Simple Youtube Agent 

A simple agent that takes Youtube video as input and answers the questions about it.

### Installations
To install the requirements run the command: 

```bash
pip install -r requirements.txt
```

### Environment Variables
To Set up the environment variables with your key, you can create a `.env` file and set its content to:
```text
OPENAI_API_KEY=<YOUR-KEY-HERE>
HUGGINGFACEHUB_API_TOKEN=<YOUR-KEY-HERE>
```

### Running the streamlit app
To run the app just enter the following command:
```bash
streamlit run app.py
```