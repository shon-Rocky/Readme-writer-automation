from llm import mixtral8x7b
from git import Repo
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM model
llm = mixtral8x7b()

# Prompting user for repository URL and path
repository = input("Please provide Repository URL: ")
repo_path = input("Please provide path to clone the repository: ")

# Remove the ".git" extension from repository URL
repository_without_git = repository[:-4] if repository.endswith(".git") else repository
repo_name = repository_without_git.split("/")[-1]

try:
    # Clone the repository
    if os.path.exists(os.path.join(repo_path, repo_name)):
        print(f"The repository '{repo_name}' exists in the specified path.")
    else:
        repo = Repo.clone_from(repository, to_path=repo_path)
except Exception as e:
    # Handle cloning exceptions
    if "destination path" in str(e) and "already exists and is not an empty directory" in str(e):
        print(f"Destination path '{repo_path}' already exists and is not empty.")
    else:
        raise Exception("Error occurred while cloning the repository:", str(e))

# Configuring document loader
loader = GenericLoader.from_filesystem(
    path=repo_path,
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(
        language=Language.PYTHON,
        parser_threshold=500
    )
)

# Loading documents using the configured loader
documents = loader.load()

# Splitting Python documents
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=2000, 
    chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

# Creating Chroma vector store
db = Chroma.from_documents(texts, GPT4AllEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

# Configuring prompt for search query generation
prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

# Creating retrieval chain with LLM
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Configuring prompt for answering user questions
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

# Creating QA chain
qa = create_retrieval_chain(retriever_chain, document_chain)

# Prompting user for directory path to save readme file
directory_path = input("Please provide directory path to save the readme file: ")

# Creating directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Writing readme file
path_to_readme = os.path.join(directory_path, "readme.md")
question = "Can you write a readme.md file for this code?"
result = qa.invoke({"input": question})
readme_content = result["answer"]

# Writing content to readme file
with open(path_to_readme, "w") as readme_file:
    readme_file.write(readme_content)

print(f"Readme file successfully written in the {path_to_readme}")
