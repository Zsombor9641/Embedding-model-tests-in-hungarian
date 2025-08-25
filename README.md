# Hungarian Embedding Models Evaluation 🇭🇺

A comprehensive evaluation framework for testing embedding models on Hungarian text data, specifically designed for the Clearservice company's Q&A dataset.

## 📋 Overview

This project evaluates multiple embedding models for Hungarian text processing, comparing their performance on retrieval tasks using a custom dataset. The evaluation includes both local and API-based models, measuring accuracy, speed, and efficiency metrics.

## 🎯 Key Features

- **Multi-model Support**: Evaluate 7+ different embedding models
- **Comprehensive Metrics**: MRR, Recall@1, Recall@3, Query Speed (QPS)
- **Detailed Timing Analysis**: Build time, search time, per-query performance
- **Visual Analytics**: 6 different visualization types
- **ChromaDB Integration**: Efficient vector database for similarity search
- **Hungarian-specific Models**: Special focus on Hungarian language processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GPU recommended for local models
- API keys (optional) for OpenAI, Gemini
- Ollama installed (optional) for local LLM embeddings

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd hungarian-embedding-evaluation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional for API models):
```bash
# Create .env file
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### Usage

1. **Run Full Evaluation**:
```bash
python main.py
```

2. **Generate Visualizations**:
```bash
python plots.py
```

3. **Individual Model Testing**: Modify `main.py` to test specific models

## 📊 Supported Models

### Local Models
- **HuBERT Hungarian**: `NYTK/sentence-transformers-experimental-hubert-hungarian`
- **BGE-M3**: `BAAI/bge-m3` (Multilingual)
- **SentenceTransformer**: `paraphrase-multilingual-MiniLM-L12-v2`

### API Models
- **OpenAI**: `text-embedding-ada-002`, `text-embedding-3-small`
- **Gemini**: `models/embedding-001`

### Ollama Models (Local LLM)
- **Nomic**: `nomic-embed-text:latest`
- **MiniLM**: `all-minilm:latest`

## 📁 Project Structure

```
hungarian-embedding-evaluation/
├── data/
│   ├── cs_qa.csv           # Q&A dataset (50 questions)
│   └── topics.txt          # Company knowledge base
├── main.py                 # Main evaluation script
├── models.py               # Model wrapper classes
├── plots.py                # Visualization generators
├── .env                    # API keys (create this)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Model Configuration
Edit `models.py` to:
- Add new embedding models
- Modify model parameters
- Implement custom embedding logic

### Evaluation Parameters
Modify `main.py` for:
- Different test datasets
- Custom evaluation metrics
- Batch processing settings

## 📈 Results Overview

Based on our evaluation of Hungarian text processing:

| Model | MRR | Recall@1 | Recall@3 | QPS | Efficiency |
|-------|-----|----------|----------|-----|------------|
| BGE-M3 | 0.903 | 0.86 | 0.96 | 6.1 | 5.5 |
| SentenceTransformer | 0.840 | 0.78 | 0.92 | 40.0 | 33.9 |
| OpenAI Ada-002 | 0.800 | 0.72 | 0.90 | 3.2 | 2.6 |
| Ollama Nomic | 0.710 | 0.64 | 0.80 | 16.2 | 11.5 |
| HuBERT Hungarian | 0.480 | 0.38 | 0.60 | 18.3 | 8.8 |

### Key Findings
- **🏆 Best Accuracy**: BGE-M3 (90.3% MRR)
- **⚡ Fastest**: SentenceTransformer Multilingual (40 QPS)
- **🎯 Best Balance**: SentenceTransformer Multilingual
- **🇭🇺 Hungarian-specific**: HuBERT needs improvement

## 📊 Visualization Types

The project generates 6 comprehensive visualizations:

1. **Performance vs Speed Scatter Plot**: Optimal model selection
2. **Comprehensive Metrics Bar Chart**: All metrics comparison
3. **Timing Breakdown**: Build vs search time analysis
4. **Efficiency Ranking**: Performance per unit time
5. **Query Speed Comparison**: Throughput analysis
6. **Performance Heatmap**: Normalized metrics overview

## 🔍 Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Primary accuracy metric
- **Recall@1**: Percentage of correct top-1 retrievals
- **Recall@3**: Percentage of correct top-3 retrievals
- **QPS (Queries Per Second)**: Processing speed
- **Efficiency Score**: MRR divided by average query time

## 🛠️ Extending the Framework

### Adding New Models

1. Create a new embedder class in `models.py`:
```python
class YourModelEmbedder:
    def __init__(self, model_name="your-model"):
        # Initialize your model
        pass
    
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # Implement encoding logic
        return embeddings
```

2. Add to evaluation in `main.py`:
```python
models.append(("Your Model Name", YourModelEmbedder()))
```

### Custom Datasets

Replace `cs_qa.csv` with your dataset containing:
- `question`: Query text
- `topic`: Expected topic/category
- `answer`: Reference answer (optional)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Install appropriate PyTorch version
2. **API Rate Limits**: Check your API quotas and billing
3. **Ollama Connection**: Ensure Ollama service is running
4. **Memory Issues**: Reduce batch sizes or use smaller models

### Performance Tips

- Use GPU for local models when available
- Implement caching for repeated evaluations
- Use batch processing for large datasets
- Monitor API usage costs

## 📄 Output Files

The evaluation generates:
- `evaluation_results_with_timing.json`: Detailed results and timing
- Various visualization plots (PNG/SVG format)
- Console output with rankings and insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test with the existing dataset
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:
- Check the troubleshooting section
- Review the code documentation
- Open an issue on GitHub
- Contact the development team

## 🔮 Future Enhancements

- [ ] Multi-language support beyond Hungarian
- [ ] Real-time model comparison dashboard
- [ ] Automatic hyperparameter tuning
- [ ] Integration with more vector databases
- [ ] Custom loss functions for domain-specific tasks
- [ ] A/B testing framework for production deployments

---

**Note**: This evaluation framework is specifically designed for Hungarian text processing but can be adapted for other languages by modifying the dataset and model selection.