from flask import Flask, request, jsonify, render_template
import os
import warnings
import asyncio
import websockets
from dotenv import load_dotenv
from flask_socketio import SocketIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from the .env file
load_dotenv()

# Fetch the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")

# Initialize Flask app and SocketIO for WebSocket communication
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the GoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

# Load the PDF document and split it into chunks
loader = PyPDFLoader("conversation.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
split_docs = text_splitter.split_documents(docs)

# Initialize HuggingFace embeddings and create vector store with Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Define the system prompt for the mental health assistant
system_prompt = (
    '''You are a mental health virtual assistant named Serene, designed to support students with their emotional and mental health concerns.
    You should assist in managing stress, anxiety, and academic pressures while offering solutions, but always with empathy and kindness.'''
)

# Set up the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Memory to retain chat history and context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the question-answer chain
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# Fetch relevant documents from the vector store
def fetch_documents(input_text):
    return retriever.get_relevant_documents(input_text)

# Process the input to generate a response
def process_input(input_text):
    try:
        docs = fetch_documents(input_text)
        chat_history = memory.load_memory_variables({})["chat_history"]
        response = qa_chain.invoke({
            "input_documents": docs,
            "input": input_text,
            "context": chat_history
        })

        if isinstance(response, dict) and "output" in response:
            return response["output"]
        return str(response)

    except Exception as e:
        print(f"Error processing input: {e}")
        return "Error processing your request."

# WebSocket client to communicate with WebSocket server
async def send_data_to_websocket_server(message):
    uri = "ws://127.0.0.1:5001"  # WebSocket server address
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(message)
            response = await websocket.recv()
            return response
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        return "Error connecting to WebSocket server"

# Flask route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('input')
    print(f"Received input: {user_input}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(send_data_to_websocket_server(user_input))
    print(f"Response from WebSocket server: {response}")

    return jsonify({'response': response})

# Serve the HTML page with a form for chatting
@app.route('/')
def home():
    return render_template('index.html')

# Run the Flask app
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
