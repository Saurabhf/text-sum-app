from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langserve import add_routes
import uvicorn
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="google-t5/t5-small",
    task="summarization",
    pipeline_kwargs={"max_new_tokens": 100},
)

# 1. Create prompt template
template = """Summaraize the given text: {text}"""
prompt_template = PromptTemplate.from_template(template)

# 2. Create parser
parser = StrOutputParser()

# 3. Create chain
chain = prompt_template | llm | parser


# 4. App definition
app = FastAPI(
  title="Text Summaraization App",
  version="1.0",
  description="A simple text summary app",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)