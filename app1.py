from flask import Flask, request, jsonify
import os
import warnings
import asyncio
import websockets
from dotenv import load_dotenv
from flask_socketio import SocketIO, send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv


load_dotenv()
# Fetch the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


app = Flask(__name__)

# Initialize SocketIO for WebSocket communication
socketio = SocketIO(app, cors_allowed_origins="*")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")

# Initialize the GoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

# Load the PDF document
loader = PyPDFLoader("conversation.pdf")
data = loader.load()

# Split the document into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Initialize embeddings model (Hugging Face)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store using Chroma for document retrieval
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})


# System prompt for the mental health virtual assistant
system_prompt = (
    '''You are a mental health virtual assistant like a psychiatrist named Serene, designed to support students with their emotional and mental health concerns.
    You can understand and respond with empathy while offering advice and resources based on the queries.
    You should assist in helping students manage stress, anxiety, academic pressures, and any other emotional or mental health concerns.
    Use your knowledge and sense to understand emotions, maintain a comforting tone, and offer solutions based on the context but be empathetic and kind at all times.
    When needed, retrieve helpful resources or suggestions from the knowledge base for students to explore.
    '''
    "{context}"
)

# Creating the prompt with system message and user input
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

# Create the question-answer chain (does not use retriever directly)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Fetch relevant documents using the retriever separately
def fetch_documents(input_text):
    # Retrieve relevant documents from the knowledge base
    docs = retriever.get_relevant_documents(input_text)
    return docs

# Process the user input and generate a response
def process_input(input_text):
    try:
        # Step 1: Fetch documents using the retriever
        docs = fetch_documents(input_text)

        # Step 2: Prepare the input with chat history (context) and user input
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Step 3: Use question-answering chain to process the retrieved documents
        response = question_answer_chain.invoke({
            "input_documents": docs,
            "input": input_text,
            "context": chat_history
        })

        # Debugging: Print the full response to see its structure
        print("Full response:", response)

        # Step 4: Handle the response based on its actual structure
        if isinstance(response, dict) and "output" in response:
            return response["output"]
        else:
            # If it's not a dict, return the raw response or handle it accordingly
            return str(response)

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request."

# Example usage
#if __name__ == "__main__":
 #   while True:
        # Prompt the user for input
  #      input_query = input("You: ")

        # Exit the loop if the user types 'exit'
   #     if input_query.lower() == 'exit':
    #        print("Exiting the chatbot. Goodbye!")
     #       break

        # Process the user input and get the response
      #  answer = process_input(input_query)

        # Print the chatbot's response
       # print("Chatbot response:", answer)
async def send_data_to_websocket_server(message):
    uri = "ws://127.0.0.1:5000"  # WebSocket server address
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(message)  # Send data to the WebSocket server
            response = await websocket.recv()  # Receive response from server
            print(f"Received from WebSocket server: {response}")
            return response
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        return "Error connecting to WebSocket server"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Get the JSON data from the request
    user_input = data.get('input')  # Extract the user input

    # Process the user input and get the response
    answer = process_input(user_input)

    # Return the response as JSON
    return jsonify({'response': answer})

@socketio.on('message')
def handle_message(message):
    print(f"Received message: {message}")  # Log the received message
    response = process_input(message)  # Process the input
    print(f"Response: {response}")  # Log the response
    send(response)

# Serve the HTML form
@app.route('/')
def home():
    return app.send_static_file('index.html')  # Serve the HTML file

#@socketio.on('connect')
# def handle_connect():
#    print("Client connected")
#    send("Welcome to the WebSocket connection!")

#@socketio.on('disconnect')
# def handle_disconnect():
#    print("Client disconnected")

#@socketio.on('message')
# def handle_message(message):
#    print(f"Received message: {message}")

if __name__ == "__main__":
    socketio.run(app, debug=True)