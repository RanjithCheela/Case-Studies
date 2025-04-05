from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from google import genai
from rich.markdown import Markdown
from rich.console import Console
import warnings

warnings.simplefilter("ignore")

#####


# load key
with open("api_keys/OpenAI_API_Key.txt", "r") as file:
    OPENAI_API_KEY = file.read().strip()

with open("api_keys/Gemini_API_Key.txt", "r") as file:
    GEMINI_API_KEY = file.read().strip()

# declare db directory
chroma_path = "chroma_db"

# declare embeding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# load the chroma db
db = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)


# create function to get the query results
def get_query_results(query_text: str, n_results=3):
    results = db.similarity_search_with_relevance_scores(query_text, n_results)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results."

    combine_results = "\n\n----\n\n".join([doc.page_content for doc, score in results])
    return combine_results


# define promt template
PROMPT_TEMPLATE = """
Answer the question based only on the following documents. Do not use any external information.:
{context}

------
Now, based on the above documents, answer without mentioning about the documents: {question}
"""


# create function to generate answer
def generate_answer(prompt_text: str):
    # create client
    client = genai.Client(api_key=GEMINI_API_KEY)
    # specify model
    model = "gemini-2.0-flash-thinking-exp-01-21"
    # generate answer
    response = client.models.generate_content(model=model, contents=prompt_text)
    return response


def main():
    console = Console()

    while True:
        # get user query
        query_text = input("\nEnter your query (type 'exit' to quit): ").strip()

        # Check for exit condition
        if query_text.lower() == "exit":
            print("\nGoodbye!")
            break

        # look up for context
        context_text = get_query_results(query_text, 4)

        # format the context to the prompt template
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # parse the context and query to the prompt template
        formatted_prompt = prompt_template.format(
            context=context_text, question=query_text
        )

        # feed the formatted query and context for the model to generate the answer
        response = generate_answer(formatted_prompt)

        # print the response
        console.print("\nHere is your answer:")
        console.print(Markdown(response.text))


main()
