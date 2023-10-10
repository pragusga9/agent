import os, openai
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI


llmc = AzureChatOpenAI(
            openai_api_base=openai.api_base,
            openai_api_key=openai.api_key,
            openai_api_type='azure',
            openai_api_version='2023-03-15-preview',
            deployment_name='gpt-35-turbo',
            model='gpt-35-turbo',
            # streaming=True,
            temperature=0
            )

llm = AzureOpenAI(
            openai_api_base=openai.api_base,
            openai_api_key=openai.api_key,
            openai_api_type='azure',
            openai_api_version='2023-03-15-preview',
            deployment_name='gpt-35-turbo',
            model='gpt-35-turbo',
            temperature=0
            )
