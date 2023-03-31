import langchain
import openai
import os
from langchain import OpenAI
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import magic
import nltk
import time
import re
import gradio as gr
from fastapi import FastAPI
import argparse

parser = argparse.ArgumentParser(description='TLDR Bot')

parser.add_argument('-k', '--key', type=str, required=True,
                    help='OpenAI API Key, format = sk-XXZXXXXXXXXXXXXXX')
args = parser.parse_args()

OPENAI_API_KEY = args.key

# Paste your OpenAI API key in the file OPENAI_API_KEY.txt (format: sk-XXZXXXXXXXXXXXXXX)

if not OPENAI_API_KEY:
  with open('OPENAI_API_KEY.txt') as f:
    OPENAI_API_KEY = f.read().strip()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

TLDR_APP_PATH = "/tldr-bot"

tldr_app = FastAPI()

source_requests = [
                   'source?',
                   'source',
                   ]

class Engine:
  title: str = ""
  qa = None
  source_document = None

  def setup_file(self, filepath):
    nltk.download('averaged_perceptron_tagger')
    try:
      loader = DirectoryLoader('/', glob = filepath[1:])
      documents = loader.load()
      text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
      texts = text_splitter.split_documents(documents)
      embeddings = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))
      docsearch = Chroma.from_documents(texts, embeddings)
      chain_type_kwargs = {
          "memory": ConversationBufferMemory()
      }
      llm = ChatOpenAI(temperature = 0, 
                   verbose = True)
      self.qa = RetrievalQA.from_chain_type(llm = llm, 
                                           chain_type = 'stuff', 
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs = chain_type_kwargs, 
                                           return_source_documents = True)
      self.title = filepath
    except:
      raise Exception("Something went wrong when processing the txt file")  

engine = Engine()

def add_text(history, text):
    history = history + [(text, None)]
    return history

def add_file(history, file):
    engine.setup_file(file.name)
    history = history + [("File succesfully uploaded. Prompt away! ✅", None)]
    return history

def bot(history, text):
    if engine.qa:
      response = engine.qa({'query': text})
      if text.lower() in source_requests:
        history[-1][1] = engine.source_document
        yield history, ""
      else:
        history[-1][1] = ""
        for info in re.split("(,|[\n\s+])", response['result']):
          history[-1][1] += info
          time.sleep(0.075)
          engine.source_document = "\"" + response['source_documents'][0].page_content + " \""
          yield history, ""
    else:
      history[-1][1] = "Upload a document first"
      yield history, ""
      
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot", label = "TLDR the T&C").style(height = 750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Upload a T&C file (pdf or txt), then enter prompt",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton(f"📄", file_types=["text", "pdf"])

    txt.submit(add_text, [chatbot, txt], [chatbot]).then(
        bot, [chatbot, txt], [chatbot, txt]
    )
    btn.upload(add_file, [chatbot, btn], [chatbot])

demo.queue()
tldr_app = gr.mount_gradio_app(tldr_app, demo, path=TLDR_APP_PATH)

@tldr_app.get("/")
def root():
   return {
        "response": "Welcome to the TLDR bot API"
   }

if __name__ == '__main__':

  import uvicorn
  uvicorn.run(tldr_app, host='0.0.0.0', port=10000)
