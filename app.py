import os
import getpass
from flask import Flask, request, jsonify, render_template
import threading
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variable for Google API Key
try:
   os.environ["GOOGLE_API_KEY"] = "AIzaSyASyu6lT7Zeu0OTM51OqsNbBpA65OXJSZY"
   api_key = os.environ["GOOGLE_API_KEY"]
except Exception as e:
    print(f"An error occurred: {e}")

# Initialize Flask app
app = Flask(__name__)

# System prompt for the mental health assistant
system_prompt = (
    '''You are a mental health virtual assistant, Alexa, designed to support students with their emotional and mental health concerns.
    You can understand and respond with empathy while offering advice and resources based on the queries.
    You should assist in helping students manage stress, anxiety, academic pressures, and any other emotional or mental health concerns.
    Use your knowledge and sense to understand emotions, maintain a comforting tone, and offer solutions based on the context but be empathetic and kind at all times.
    When needed, retrieve helpful resources or suggestions from the knowledge base for students to explore.
    '''
    "{context}"
)
# Create the prompt template with system message and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Memory for chat history retention and context
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Store conversation history
    return_messages=True        # Return previous messages as part of the context
)

# Initialize Google Generative AI (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Embedding and vectorstore initialization (Make sure `docs` are defined)
# Example: docs = ... (Load or create your documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# Create retriever from the vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Function to fetch documents using the retriever
def fetch_documents(input_text):
    docs = retriever.get_relevant_documents(input_text)
    return docs
# Process input to generate a response
def process_input(input_text):
    try:
        # Fetch relevant documents
        docs = fetch_documents(input_text)

        # Retrieve chat history
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Generate the response using the chain
        response = question_answer_chain.invoke({
            "input_documents": docs,
            "input": input_text,
            "context": chat_history
        })

        # Handle and return the response
        if isinstance(response, dict) and "output" in response:
            return response["output"]
        else:
            return str(response)

    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request right now."

# Define an API endpoint
@app.route('/')
def index():
    return render_template('index.html')

def process_input(input_text):
    try:
        # Fetch relevant documents (mocked here for simplicity)
        docs = ["This is a mock response."]
        # Generate the response
        response = f"You asked: {input_text}. Here is a helpful response."
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request right now."

# Define the API endpoint for the chatbot
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('input')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Process input and generate a response
    response = process_input(user_input)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
