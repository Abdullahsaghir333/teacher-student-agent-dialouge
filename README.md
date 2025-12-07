# 🎓 AI Teacher-Student Agent Dialogue System

A comprehensive AI-powered educational platform that simulates interactive classroom dialogues between AI teacher and student agents, featuring RAG (Retrieval-Augmented Generation), quiz generation, lesson summaries, and comprehensive model evaluation capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)
- [Technologies Used](#technologies-used)
- [Model Evaluation](#model-evaluation)
- [Docker Support](#docker-support)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an intelligent AI classroom system that demonstrates the effectiveness of RAG (Retrieval-Augmented Generation) in educational contexts. The system features:

- **Interactive Dialogue System**: AI teacher and student agents engage in natural, educational conversations
- **RAG Integration**: Teacher agent uses RAG to provide contextually accurate responses from a knowledge base
- **Quiz Generation**: Automatically generates quizzes based on lesson transcripts
- **Lesson Summaries**: Creates structured summaries of teaching sessions
- **Model Evaluation**: Comprehensive evaluation framework comparing multiple LLMs (GPT-4o-mini, LLaMA3-8B, DistilGPT-2) across different settings (Zero-Shot, Few-Shot, RAG)
- **Streamlit UI**: User-friendly web interface for interacting with the system

---

## ✨ Features

### Core Features

1. **AI Teacher-Student Dialogue**
   - Natural conversation flow between AI agents
   - Configurable dialogue turns (1-10)
   - Topic-based learning sessions
   - Real-time conversation display

2. **RAG-Powered Teaching**
   - FAISS vector database for efficient retrieval
   - HuggingFace embeddings (all-MiniLM-L6-v2)
   - Context-aware responses using retrieved knowledge
   - Comparison between RAG and non-RAG teaching modes

3. **Quiz Generation**
   - Automatic MCQ generation from lesson transcripts
   - 5 questions per quiz
   - Interactive quiz interface
   - Automatic scoring system

4. **Lesson Summarization**
   - Structured summaries of teaching sessions
   - Topic-focused content extraction
   - Easy-to-read format

5. **Model Evaluation Framework**
   - Multi-model comparison (GPT-4o-mini, LLaMA3-8B, DistilGPT-2)
   - Multiple evaluation settings (Zero-Shot, Few-Shot, RAG)
   - Comprehensive metrics (BLEU, ROUGE-L, BERTScore, Semantic Similarity)
   - Visualization of evaluation results

6. **Data Processing & Visualization**
   - ScienceQA dataset integration
   - Data preprocessing and analysis
   - Statistical visualizations

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                    │
│  (main.py - User Interface & Session Management)        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              LangGraph State Machine                     │
│  (classroom_graph.py - Dialogue Orchestration)          │
└──────┬──────────────────────────────────────┬───────────┘
       │                                      │
┌──────▼──────────┐              ┌───────────▼──────────┐
│  Student Agent  │              │   Teacher Agent       │
│  (student.py)   │◄────────────►│   (teacher.py)       │
└─────────────────┘              └───────────┬───────────┘
                                            │
                                   ┌────────▼──────────┐
                                   │   RAG Retriever   │
                                   │  (retriever.py)   │
                                   └────────┬──────────┘
                                            │
                                   ┌────────▼──────────┐
                                   │  FAISS Vector DB  │
                                   │   (loader.py)     │
                                   └───────────────────┘
```

### Data Flow

1. **User Input**: Topic and dialogue turns via Streamlit sidebar
2. **RAG Index Loading**: FAISS vector store loaded with embeddings
3. **Dialogue Loop**: 
   - Student agent asks question based on teacher's previous response
   - Teacher agent retrieves relevant context using RAG
   - Teacher agent generates answer using retrieved context
   - Conversation stored in session state
4. **Post-Processing**: 
   - Transcript generation
   - Quiz generation (optional)
   - Summary generation (optional)

---

## 📁 Project Structure

```
teacher-student-agent-dialouge/
│
├── main.py                      # Streamlit UI and main application entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration for containerization
├── .env                         # Environment variables (API keys)
│
├── agents/                      # AI Agent Implementations
│   ├── student.py              # Student agent - generates questions
│   ├── teacher.py              # Teacher agent - provides answers
│   └── evaluator.py            # Evaluation agent for session quality
│
├── graph/                       # LangGraph State Machine
│   └── classroom_graph.py      # Dialogue orchestration graph
│
├── rag/                         # RAG Implementation
│   ├── loader.py               # FAISS index loading
│   └── retriever.py            # Context retrieval from vector DB
│
├── prompts/                     # Prompt Templates
│   ├── student_prompt.py       # Student agent prompts
│   └── teacher_prompt.py       # Teacher agent prompts
│
├── core/                        # Core Utilities
│   ├── dialogue_manager.py     # Dialogue session management
│   └── topic_selector.py       # Topic selection utility
│
├── utils/                       # Utility Functions
│   └── config.py               # Configuration management
│
├── data_processor.py            # Dataset loading and preprocessing
├── embedding.py                 # FAISS index creation script
├── model_evaluation.py          # Model evaluation framework
│
└── faiss_index_fast/            # Generated FAISS vector database (not in repo)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12.6 or higher
- pip package manager
- OpenAI API key
- (Optional) Groq API key for LLaMA3-8B evaluation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd teacher-student-agent-dialouge
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here  # Optional, for model evaluation
```

### Step 5: Build RAG Index

Before running the application, you need to create the FAISS vector index:

```bash
python embedding.py
```

This will:
- Load the Alpaca dataset from HuggingFace
- Convert examples to dialogue turns
- Create 64-turn chunks with 16-turn overlap
- Build FAISS index using MiniLM embeddings
- Save index to `faiss_index_fast/` directory

**Note**: This process may take several minutes depending on your system.

---

## ⚙️ Configuration

### Application Settings

The application can be configured through:

1. **Streamlit Sidebar** (Runtime):
   - Topic selection
   - Number of dialogue turns (1-10)

2. **Environment Variables** (`.env`):
   - `OPENAI_API_KEY`: Required for LLM operations
   - `GROQ_API_KEY`: Optional, for LLaMA3-8B evaluation

3. **Code Configuration**:
   - Model selection: Edit `main.py` to change LLM models
   - Temperature settings: Adjust in agent initialization
   - RAG retrieval count: Modify `k` parameter in `rag/retriever.py`

### Model Configuration

Default models used:
- **Student Agent**: GPT-4o-mini (temperature: 0.7)
- **Teacher Agent**: GPT-4o-mini (temperature: 0.3)
- **Quiz Generator**: GPT-4o-mini (temperature: 0.2)
- **Summary Generator**: GPT-4o-mini (temperature: 0.3)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2

---

## 📖 Usage

### Running the Application

1. **Start Streamlit App**:

```bash
streamlit run main.py
```

2. **Access the UI**:
   - Open your browser to `http://localhost:8501`
   - The application interface will load

### Using the Application

1. **Start a Lesson**:
   - Enter a topic in the sidebar (e.g., "Quantum Computing")
   - Select number of dialogue turns (1-10)
   - Click "Start Lesson" button

2. **View Conversation**:
   - Watch the dialogue unfold in real-time
   - Student questions and teacher answers appear sequentially
   - Full transcript displayed below

3. **Generate Quiz**:
   - Click "Generate Quiz" button
   - Review the generated MCQ questions
   - Select your answers
   - Click "Submit Answers" to see your score

4. **Generate Summary**:
   - Click "Generate Summary" button
   - View structured lesson summary in text area

### Running Model Evaluation

To run the comprehensive model evaluation:

```bash
python model_evaluation.py
```

This will:
- Compare GPT-4o-mini, LLaMA3-8B, and DistilGPT-2
- Test Zero-Shot, Few-Shot, and RAG settings
- Compute BLEU, ROUGE-L, BERTScore, and Semantic Similarity metrics
- Generate visualization plots
- Save results to `evaluation_outputs/` directory

### Processing Dataset

To process and visualize the ScienceQA dataset:

```bash
python data_processor.py
```

This will:
- Load ScienceQA dataset from HuggingFace
- Preprocess and analyze data
- Generate visualization plots
- Save visualization to `dataset_visualization.png`

---

## 🔧 Components

### 1. Student Agent (`agents/student.py`)

**Purpose**: Simulates a curious student asking follow-up questions.

**Key Features**:
- Generates contextually relevant questions
- Uses previous teacher response as context
- Maintains topic focus throughout dialogue

**Configuration**:
- Model: GPT-4o-mini
- Temperature: 0.7 (for creativity)
- Prompt: Encourages short, focused questions

### 2. Teacher Agent (`agents/teacher.py`)

**Purpose**: Provides educational answers using RAG-retrieved context.

**Key Features**:
- Retrieves relevant context from FAISS index
- Generates structured, educational responses
- Can operate in RAG or non-RAG mode

**Configuration**:
- Model: GPT-4o-mini
- Temperature: 0.3 (for accuracy)
- Context integration: RAG-retrieved knowledge

### 3. RAG System (`rag/`)

**Components**:
- **Loader** (`loader.py`): Loads pre-built FAISS index
- **Retriever** (`retriever.py`): Performs similarity search and context retrieval

**Features**:
- FAISS vector database for fast similarity search
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Configurable retrieval count (default: k=2)
- Returns concatenated context from top-k documents

### 4. Classroom Graph (`graph/classroom_graph.py`)

**Purpose**: Orchestrates dialogue flow using LangGraph.

**State Management**:
- Topic: Current learning topic
- Last teacher reply: Previous teacher response
- Last student question: Current student question
- Conversation: List of (role, message) tuples

**Graph Structure**:
```
Entry → Student Turn → Teacher Turn → End
```

### 5. Quiz Generator (`main.py`)

**Features**:
- Generates exactly 5 MCQ questions
- Based on lesson transcript and topic
- Parses questions, options, and answers
- Interactive quiz interface with scoring

### 6. Summary Generator (`main.py`)

**Features**:
- Creates structured lesson summaries
- Topic-focused content extraction
- Readable format for review

### 7. Model Evaluation (`model_evaluation.py`)

**Capabilities**:
- Multi-model comparison
- Multiple evaluation settings
- Comprehensive metrics
- Visualization generation

**Metrics Computed**:
- **BLEU**: N-gram overlap score
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Contextual embedding similarity
- **Semantic Similarity**: Cosine similarity of embeddings

---

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.12.6**: Programming language
- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration framework
- **LangGraph**: State machine for dialogue management
- **OpenAI API**: GPT-4o-mini model access

### ML/AI Libraries

- **FAISS**: Vector similarity search
- **HuggingFace**: 
  - Transformers library
  - Sentence Transformers
  - Datasets library
- **sentence-transformers**: Embedding models

### Evaluation Libraries

- **NLTK**: BLEU score computation
- **rouge-score**: ROUGE metrics
- **bert-score**: BERTScore evaluation
- **scikit-learn**: Additional ML utilities

### Data Processing

- **Pandas**: Data manipulation
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualizations

### Other Libraries

- **python-dotenv**: Environment variable management
- **pydantic**: Data validation
- **tqdm**: Progress bars

---

## 📊 Model Evaluation

### Evaluation Framework

The project includes a comprehensive evaluation framework that compares:

**Models**:
1. GPT-4o-mini (OpenAI)
2. LLaMA3-8B (Groq API)
3. DistilGPT-2 (Local/HuggingFace)

**Settings**:
1. **Zero-Shot**: Direct prompt without examples
2. **Few-Shot**: Prompt with example demonstrations
3. **RAG**: Context retrieved from FAISS index

### Metrics

1. **BLEU Score**: Measures n-gram precision
2. **ROUGE-L**: Measures longest common subsequence
3. **BERTScore**: Contextual semantic similarity
4. **Semantic Similarity**: Cosine similarity of embeddings

### Results Storage

Evaluation results are saved in `evaluation_outputs/`:
- `raw_outputs.json`: Model outputs for each setting
- `metrics.json`: Computed metrics
- `*.png`: Visualization plots for each metric

---

## 🐳 Docker Support

### Building Docker Image

```bash
docker build -t ai-classroom .
```

### Running Container

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key_here \
  -e GROQ_API_KEY=your_key_here \
  ai-classroom
```

### Dockerfile Features

- Python 3.12.6 slim base image
- CPU-only PyTorch installation
- All dependencies pre-installed
- Streamlit server on port 8501
- Health check endpoint

**Note**: Ensure FAISS index is built before containerizing, or include index building in Dockerfile.

---

## 🔮 Future Enhancements

### Planned Features

1. **Multi-Topic Support**
   - Pre-defined topic library
   - Topic difficulty levels
   - Adaptive learning paths

2. **Enhanced Evaluation**
   - Human evaluation integration
   - More evaluation metrics
   - Comparative analysis dashboard

3. **Advanced RAG**
   - Multiple knowledge sources
   - Hybrid search (keyword + semantic)
   - Query expansion

4. **Student Personalization**
   - Learning style adaptation
   - Difficulty adjustment
   - Progress tracking

5. **UI Improvements**
   - Real-time conversation streaming
   - Export transcripts/quiz results
   - Session history

6. **Performance Optimization**
   - Caching mechanisms
   - Async processing
   - Batch operations

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions/classes
- Include tests for new features
- Update README if adding new features
- Ensure all tests pass before submitting

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **OpenAI**: For GPT-4o-mini API
- **HuggingFace**: For datasets, models, and embeddings
- **LangChain**: For LLM orchestration framework
- **Streamlit**: For web UI framework
- **Alpaca Dataset**: For RAG knowledge base
- **ScienceQA Dataset**: For evaluation data

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the project maintainers

---

## 🔍 Troubleshooting

### Common Issues

1. **FAISS Index Not Found**
   - Solution: Run `python embedding.py` to create the index

2. **OpenAI API Key Error**
   - Solution: Ensure `.env` file exists with valid `OPENAI_API_KEY`

3. **Import Errors**
   - Solution: Ensure virtual environment is activated and all dependencies are installed

4. **Port Already in Use**
   - Solution: Change Streamlit port: `streamlit run main.py --server.port 8502`

5. **Memory Issues**
   - Solution: Reduce chunk size in `embedding.py` or use smaller models

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**Last Updated**: 2024
**Version**: 1.0.0

