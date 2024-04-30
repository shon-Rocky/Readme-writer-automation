from src.llm_model import llm
from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo
import os
import shutil
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate




# Initialize the LLM model
llm = llm()



# Prompting user for repository URL and path
repository = input("Please provide Repository URL: ")
repo_path = input("Please provide path to clone the repository in you computer: ")

# Remove the ".git" extension from repository URL
repository_without_git = repository[:-4] if repository.endswith(".git") else repository
repo_name = repository_without_git.split("/")[-1]

try:
    # Clone the repository
    repo_dir = os.path.join(repo_path, repo_name)
    if os.path.exists(repo_dir):
        print(f"The repository '{repo_name}' exists in the specified path.")
    else:
        try:
            repo = Repo.clone_from(repository, to_path=repo_dir)
        except InvalidGitRepositoryError as e:
            print(f"Error: The specified repository URL '{repository}' is invalid.")
            print(str(e))
            exit()  # Exit the script if the repository URL is invalid
except InvalidGitRepositoryError as e:
    print(f"Error: The specified repository URL '{repository}' is invalid.")
    print(str(e))
except NoSuchPathError as e:
    print(f"Error: The specified destination path '{repo_path}' does not exist.")
    print(str(e))
except GitCommandError as e:
    if "destination path already exists" in str(e):
        print(f"Warning: The destination path '{repo_dir}' already exists and is not empty.")
        choice = input(f"Do you want to remove the existing contents of '{repo_dir}' and continue cloning? (y/n) ")
        if choice.lower() == "y":
            shutil.rmtree(repo_dir)
            repo = Repo.clone_from(repository, to_path=repo_dir)
            print("Repository cloned successfully.")
        else:
            print("Cloning process canceled.")
    else:
        print("Error occurred while cloning the repository:")
        print(str(e))
except Exception as e:
    print("An unexpected error occurred:")
    print(str(e))

# Configuring document loader
loader = GenericLoader.from_filesystem(
    path=repo_dir,
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
    chunk_size=1000, 
    chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

# Creating Chroma vector store
db = Chroma.from_documents(texts, GPT4AllEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

# Configuring prompt for answering user questions
prompt = ChatPromptTemplate.from_template(
    """Provide a well-structured README file for a software project, including a clear title  a brife description of the project,
    table of contents, installation guide with dependencies, usage instructions,
    configuration details, contributing guidelines, testing information, deployment steps,
    list of technologies/tools used, versioning scheme, licensing, author/contributor details,
    acknowledgments, and support contacts. Use a logical structure, concise language, and visuals where appropriate.
    Adjust the sections and their content based on the specific needs of your project.
    For example, if your project doesn't have deployment or testing steps, you can remove those sections or replace them with relevant sections. 
    <context>
    {context}
    </context>
    Question: {input}"""
)
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Prompting user for directory path to save readme file
directory_path = input("Please provide directory path to save the readme file: ")

# Creating directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Writing readme file
path_to_readme = os.path.join(directory_path, "readme.md")
question = "Can you write a readme.md file for this code?"
result = retrieval_chain.invoke({"input": question})
readme_content = result["answer"]

# Writing content to readme file
with open(path_to_readme, "w") as readme_file:
    readme_file.write(readme_content)

print(f"Readme file successfully written in the {path_to_readme}")