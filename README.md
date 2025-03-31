# NEXUS: Neural-Symbolic Reasoning System

NEXUS is an advanced hybrid architecture that integrates neural networks with symbolic reasoning for more robust, interpretable, and accurate decision-making. By combining the pattern recognition capabilities of neural networks with the logical reasoning of symbolic AI, NEXUS achieves improved performance in complex domains.

## Core Concepts

- **Neural-Symbolic Integration**: Combines deep learning with logical reasoning for enhanced decision-making
- **Metacognitive Control**: Dynamically decides when to rely on neural vs. symbolic components
- **Interpretable Decisions**: Provides detailed reasoning steps explaining model predictions
- **Knowledge Representation**: Utilizes a knowledge graph for domain knowledge encoding

## Implementations

The repository provides two implementation variants:

### 1. Claude LLM Implementation (`nexus_claude.py`)

This variant uses Anthropic's Claude as the neural component:

- Leverages Claude API for neural predictions and concept extraction
- Maintains a knowledge graph for symbolic reasoning
- Combines neural and symbolic predictions through a metacognitive controller
- Excellent for text-based inputs and medical diagnosis applications

### 2. PyTorch Implementation (`nexus_real_data.py`)

A transformer-based implementation with:

- Advanced neural network with transformer architecture
- Enhanced knowledge graph for symbolic reasoning
- Neural-symbolic interface layer for translating between representations
- Comprehensive evaluation and visualization capabilities

## Getting Started

### Prerequisites

```
python >= 3.6
anthropic
httpx
numpy
torch
pandas
requests
scikit-learn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nexus.git
cd nexus

# Install dependencies
pip install -r requirements.txt

# Set up your API key for Claude (for nexus_claude.py)
export CLAUDE_API_KEY="your-api-key"
```

## Usage Examples

### Using the Claude Implementation

```python
import os
from nexus_claude3 import ClaudeLLM, EnhancedKnowledgeGraph, NEXUS_Claude, run_nexus

# Initialize through the main function
nexus, results = run_nexus()

# Or create a custom instance
api_key = os.environ.get("CLAUDE_API_KEY")
claude = ClaudeLLM(api_key=api_key)

# Define your knowledge graph
knowledge_graph = EnhancedKnowledgeGraph()
# Add entities, relations, and rules to the knowledge graph
knowledge_graph.add_entity(0, "symptom_1")
knowledge_graph.add_entity(1, "symptom_2")
knowledge_graph.add_entity(100, "heart_disease")
knowledge_graph.add_relation(0, "indicates", 100, weight=0.8)

# Initialize NEXUS with Claude
nexus = NEXUS_Claude(
    claude_llm=claude,
    knowledge_graph=knowledge_graph,
    class_names=["No Heart Disease", "Heart Disease"],
    symbol_names=["symptom_1", "symptom_2"]
)

# Make a diagnosis
patient_description = "65-year-old male with chest pain and elevated blood pressure"
diagnosis = nexus.diagnose(patient_description)

# Get explanation
explanation = nexus.explain_diagnosis(diagnosis, detail_level='high')
print(explanation)
```

### Using the PyTorch Implementation

```python
import torch
from nexus_real_data import run_nexus_experiment_real_data

# Run a full experiment
results = run_nexus_experiment_real_data(
    dataset_name="your-dataset-name",
    max_samples=10000,
    num_epochs=10,
    batch_size=128,
    learning_rate=0.001,
    output_dir="results",
    device="cuda" if torch.cuda.is_available() else "cpu",
    random_state=42
)

# Access the trained model
model = results['model']

# Make a prediction
input_data = torch.tensor([...])  # Feature vector
diagnosis = model.diagnose(input_data)

# Get detailed explanation
print(model.explain_diagnosis(diagnosis, detail_level='high'))

# Visualize evaluation results
model.visualize_results(output_prefix="experiment1", save_figures=True)
```

## Applications

NEXUS has demonstrated strong performance in several domains:

- **Medical Diagnosis**: Primarily designed for heart disease prediction with interpretable reasoning
- **Complex Decision Support**: Applicable to domains requiring both pattern recognition and logical reasoning
- **Interpretable AI Systems**: When transparency and explainability are required alongside high performance

## Key Components

### Enhanced Knowledge Graph

A flexible symbolic reasoning engine featuring:

- Entity representation with attributes
- Weighted relations between entities
- Logical rules with confidence scores
- Hierarchical relationships
- Multi-hop reasoning capability

### Metacognitive Controller

Intelligently combines predictions from neural and symbolic components:

- Adapts thresholds based on confidence levels
- Considers risk levels for different scenarios
- Maintains strategy history for analysis
- Learns from past decisions

### Neural Models

Two options available:

1. **Claude LLM**: Uses Claude API for concept extraction and natural language understanding
2. **Transformer Model**: Custom PyTorch implementation with attention mechanisms

## Evaluation

NEXUS provides comprehensive evaluation tools:

- Accuracy comparisons between neural, symbolic, and hybrid approaches
- Confusion matrices and F1 scores by class
- Analysis of model agreement and disagreement cases
- Confidence distribution visualization
- Strategy usage statistics

## License

This project is available under the MIT License.

## Citation

If you use NEXUS in your research, please cite:

```
@article{nexus2025,
  title={NEXUS: A Neural-Symbolic Architecture for Robust and Interpretable AI},
  author={NEXUS Team},
  year={2025}
}
```

## Acknowledgements

- UCI Heart Disease Dataset
- Anthropic's Claude for the LLM implementation
- PyTorch team for the deep learning framework