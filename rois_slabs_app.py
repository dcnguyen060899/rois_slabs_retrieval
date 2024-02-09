# dependencies
"""
import streamlit as st
import os
import sys
import openai
# initialize open ai agent model
openai.api_key = st.secrets["openai_api_key"]
# os.environ["ACTIVELOOP_TOKEN"] = ''
# Fetching secrets
os.environ['ACTIVELOOP_TOKEN'] = st.secrets["active_loop_token"]
# %%
# Imports
#
from typing import List
from llama_hub.tools.weather import OpenWeatherMapToolSpec
from llama_index import (
    Document,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.output_parsers import PydanticOutputParser
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.vector_stores import DeepLakeVectorStore
from pydantic import BaseModel
from llama_index.readers.deeplake import DeepLakeReader
import random
from llama_index.storage.storage_context import StorageContext
from typing import List, Tuple
import deeplake
from PIL import Image
from io import BytesIO
import re
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
# import cv2
import pandas as pd
import ipywidgets as widgets
from llama_index import set_global_service_context
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from llama_index import set_global_service_context

# retrieve all the image_ids we have in the folder
class roi_slabs(BaseModel):
    """Data model for image_id in a specific folder"""

    slabs: str
    rois: str


llm = OpenAI(language_model='gpt-4', temperature=.7)

reader = DeepLakeReader()
query_vector = [random.random() for _ in range(1536)]
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://dcnguyen060899/rois_slabs_db",
    limit=5,
)

dataset_path = 'rois_slabs_db'
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

index_vector_store = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    service_context=service_context)

roi_slab_engine = index_vector_store.as_query_engine(output_cls=roi_slabs)

roi_slab_engine_tool = QueryEngineTool(
    query_engine=roi_slab_engine,  # Assuming roi_slab_engine is defined elsewhere and passed here as a parameter
    metadata=ToolMetadata(
        name="roi_slab_engine",
        description=(
            "This tool is designed to translate user input into a structured JSON format, enabling the configuration and execution of database queries based on specified "
            "regions of interest (ROIs) and anatomical slabs."
            "\n\nUsage: "
            "Provide a natural language description of the desired data retrieval or analysis task, "
            "specifying anatomical locations, ROIs, and any specific measurement types (e.g., volume, "
            "cross-sectional area). The engine will ask for additional details if necessary, ensuring "
            "a comprehensive and precise JSON output for database configuration."
            "\n\nExample Input: "
            "I want to see the average at the midpoint of the third lumbar vertebra of all skeletal muscle at -29, 150 excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)"            "\n\nExample Output: "
            "{"
            "  'publish': {"
            "    'slabs': 'avg-L3mid[1]',"
            "    'rois': 'ALLSKM[-29,150]_NOARMS',"
            "    'cross_sectional_area': 'true',"
            "    'volume': 'true',"
            "    'hu': 'true'"
            "  }"
            "}"
        ),
    ),
)

agent = OpenAIAgent.from_tools(
  system_prompt = """Agents are equipped with access to a comprehensive database that contains all possible combinations of "slabs" (anatomical locations) and "rois" (regions of interest), including specific measurement ranges when applicable. This database is an essential tool for correctly interpreting user requests and translating them into the required JSON format for medical imaging analysis database queries or data processing tasks.

When a user makes a request, it is your responsibility to accurately map their description to the corresponding "slabs" and "rois" combination in the database. This includes recognizing specific anatomical features, such as "Visceral Adipose Tissue" (VAT) and measurement ranges (e.g., "-150, -50"), as well as anatomical locations like "sacrum."



Agents have access to a comprehensive database containing all possible combinations of "slabs" and "rois", which includes specific measurement ranges for a variety of tissues and organs. These are often referred to by abbreviations:


Slabs:
- FULL_SCAN: Full Body Scan
- T1, T2, T3, ... T12: Thoracic vertebrae 1 through 12
- L1, L2, L3, ... L5: Lumbar vertebrae 1 through 5 
- SACRUM: Sacrum
- avg-L3mid[1]: Average at the midpoint of the third lumbar vertebra
- t12start, t12mid, t12end: Start, mid, and end points of the twelfth thoracic vertebra
- l1start, l1mid, l1end: Start, mid, and end points of the first lumbar vertebra
- l2start, l2mid, l2end: Start, mid, and end points of the second lumbar vertebra
- l3start, l3mid, l3end: Start, mid, and end points of the third lumbar vertebra
- l4start, l4mid, l4end: Start, mid, and end points of the fourth lumbar vertebra
- l5start, l5mid, l5end: Start, mid, and end points of the fifth lumbar vertebra
- sacrumstart, sacrummid, sacrumend: Start, mid, and end points of the sacrum

ROIs:
- FULL_SCAN: Full Scan
- ALLSKM: All Skeletal Muscle
- SAT: Subcutaneous Adipose Tissue
- ALLFAT: All Fat Tissue
- ALLIMAT: All Imaging Material
- VAT: Visceral Adipose Tissue
- EpAT: Epidural Adipose Tissue
- PaAT: Paravertebral Adipose Tissue
- ThAT: Thoracic Adipose Tissue
- LIV: Liver
- SPL: Spleen
- AOC: Abdominal Oblique Composite
- CAAC: Combined Abdominal Adipose Compartment
- NOARMS: Excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)

The process for handling user requests is as follows:

1. Review the user's request to identify key terms that correspond to "slabs" and "rois" within the database. For instance, if a user asks for "Visceral adipose tissue at -150, -50" and specifies "sacrum" as the slab, you should recognize "VAT" as the roi and "[-150,-50]" as the measurement range, with "SACRUM" as the slab.

2. If the user's request is vague or lacks the specific details necessary for creating a structured JSON, engage in a clarification process. Use targeted follow-up questions to determine the precise anatomical location ("slabs") and the specific ROI ("rois"), including any measurement ranges.

3. Accurately translate the request into JSON, using the exact terminology and format from the database. Ensure "slabs" and "rois" are correctly identified and formatted, reflecting the specific combination found in the database.

Example Process for a Specific Request:
>>> User: "I want to see the average at the midpoint of the third lumbar vertebra of all skeletal muscle at -29, 150 excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)"
>>> Agent: Recognizes "SACRUM" as the slab and "VAT[-150,-50]" as the roi based on the database combinations.
>>>Final JSON Output:

{
  "publish": {
  "slabs": "avg-L3mid[1]",
  "rois": "ALLSKM[-29,150]_NOARMS",
  "cross_sectional_area": "true",
  "volume": "true",
  "hu": "true"
  }
}

Your responses should leverage the database to provide accurate and detailed structures for user requests in JSON format, ensuring a precise match between the user's description and the database's "slabs" and "rois" combinations.
Finally, the system clarifies as needed, then produces a single JSON output reflecting the database's existing "slabs" and "rois" combination.
""",
  tools=[
    roi_slab_engine_tool,
  ],
  verbose=True)

# Create the Streamlit UI components
st.title('👔 Rois-Slabs AI Gen 🧩')

# Session state for holding messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Input your prompt here')
if prompt:
   # Directly query the OpenAI Agent
   st.chat_message('user').markdown(prompt)
   st.session_state.messages.append({'role': 'user', 'content': prompt})
   response = agent.chat(prompt)
   final_response = response.response
   st.chat_message('assistant').markdown(final_response)
   st.session_state.messages.append({'role': 'assistant', 'content': final_response})
