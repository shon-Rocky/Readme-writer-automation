from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def llm(
    temperature=1,
    max_token=1024,
    top_p=0.75,
    model_id='mistralai/Mixtral-8x7B-Instruct-v0.1'
):
    """Creates a HuggingFaceEndpoint instance for the Starcoder LLM.

    Args:
        temperature: Temperature for sampling from the model (default: 0.8).
        max_token: Maximum number of tokens to generate (default: 1024).
        top_p: Nucleus sampling parameter (default: 0.75).
        model_id: ID of the Starcoder mistral on Hugging Face (default: 'mistralai/Mixtral-8x7B-Instruct-v0.1').
        add_to_git_credential: Whether to set the git credential (default: False).

    Returns:
        HuggingFaceEndpoint: An instance configured for the mistral model.
    """
    
    model = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_token,
        temperature=temperature,
        top_p=top_p,
    )


    return model
