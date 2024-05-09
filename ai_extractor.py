import os

from dotenv import load_dotenv
from langchain.chains import (create_extraction_chain,
                              create_extraction_chain_pydantic)
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                 openai_api_key=openai_api_key)


def extract(content: str, **kwargs):
    """
    The `extract` function takes in a string `content` and additional keyword arguments, and returns the
    extracted data based on the provided schema.
    """

    # This part just formats the output from a Pydantic class to a Python dictionary for easier reading. Feel free to remove or tweak this.
    if 'schema_pydantic' in kwargs:
        response = create_extraction_chain_pydantic(
            pydantic_schema=kwargs["schema_pydantic"], llm=llm).run(content)
        response_as_dict = [item.dict() for item in response]

        return response_as_dict
    else:
        return create_extraction_chain(schema=kwargs["schema"], llm=llm).run(content)


# import os

# from dotenv import load_dotenv
# from langchain.chains import (create_extraction_chain,
#                               create_extraction_chain_pydantic)
# # from langchain.chat_models import ChatOpenAI
# from langchain_community.llms import HuggingFaceHub


# load_dotenv()

# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
# hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# llm = HuggingFaceHub(temperature=0.5, repo_id="mistralai/Mistral-7B-Instruct-v0.2")


# def extract(content: str, **kwargs):
#     """
#     The `extract` function takes in a string `content` and additional keyword arguments, and returns the
#     extracted data based on the provided schema.
#     """

#     # This part just formats the output from a Pydantic class to a Python dictionary for easier reading. Feel free to remove or tweak this.
#     if 'schema_pydantic' in kwargs:
#         response = create_extraction_chain_pydantic(
#             pydantic_schema=kwargs["schema_pydantic"], llm=llm).run(content)
#         response_as_dict = [item.dict() for item in response]

#         return response_as_dict
#     else:
#         return create_extraction_chain(schema=kwargs["schema"], llm=llm).run(content)
