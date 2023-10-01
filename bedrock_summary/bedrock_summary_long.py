#
#  pip install --quiet langchain==0.0.304 "transformers>=4.24,<5"
#  mazon.titan-tg1-large が AWSアカウントで使えないので、まだ動作確認できていない
# 
import json
import os
import sys

import boto3

from utils import  print_ww

boto3_bedrock = boto3.client("bedrock-runtime")

# Configuring LangChain with Boto3

from langchain.llms.bedrock import Bedrock

llm = Bedrock(
    model_id="amazon.titan-tg1-large",
    model_kwargs={
        "maxTokenCount": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    },
    client=boto3_bedrock,
)

# Loading a text file with many tokens

shareholder_letter = "./letters/2022-letter.txt"

with open(shareholder_letter, "r") as file:
    letter = file.read()
    
llm.get_num_tokens(letter)

# Splitting the long text into chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

docs = text_splitter.create_documents([letter])

num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)

# Summarizing chunks and combining them

# Set verbose=True if you want to see the prompts being used
from langchain.chains.summarize import load_summarize_chain
summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

output = summary_chain.run(docs)

print_ww(output.strip())
