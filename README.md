# NEXUS: A Hybrid Neural-Symbolic Framework

NEXUS is an advanced neural-symbolic architecture designed to combine the strengths of neural networks with symbolic reasoning. By integrating these complementary approaches, NEXUS aims to achieve improved performance, better interpretability, and more robust decision-making.

## Features

- **Hybrid architecture**: Combines the pattern recognition capabilities of neural networks with the logical reasoning of symbolic AI
- **Metacognitive control**: Intelligently decides when to rely on neural vs. symbolic components
- **Enhanced interpretability**: Provides detailed reasoning steps and explanations for its decisions
- **Robust knowledge representation**: Utilizes a knowledge graph with entities, relations, rules, and hierarchies
- **Improved performance**: Often outperforms both standalone neural and symbolic approaches

## Implementation Variants

The repository provides two implementation variants:

### 1. PyTorch Implementation (`nexus_real_data.py`)

The PyTorch implementation includes:

- Advanced neural component with transformer architecture
- Enhanced knowledge graph for symbolic reasoning
- Neural-symbolic interface for translating between representations
- Metacognitive controller for strategic reasoning
- Comprehensive evaluation and visualization tools

### 2. Claude LLM Implementation (`nexus_claude.py`)

This version leverages the Claude Large Language Model as the neural component:

- Uses Anthropic's Claude API for neural predictions
- Maintains the knowledge graph for symbolic reasoning
- Extracts concepts from text for symbolic processing
- Metacognitive decision-making for optimal predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/alesso/nexus.git
cd nexus

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- PyTorch
- NumPy
- Anthropic API (for Claude variant)
- HTTPX
- Additional libraries for visualization (matplotlib, seaborn)

## Usage

### PyTorch Implementation

```python
import torch
from nexus_real_data import EnhancedNEXUSModel, run_nexus_experiment_real_data

# Run experiment with a dataset
results = run_nexus_experiment_real_data(
    dataset_name="your-dataset",
    max_samples=10000,
    num_epochs=10,
    batch_size=128,
    learning_rate=0.001,
    output_dir="results",
    device="cuda" if torch.cuda.is_available() else "cpu",
    random_state=42
)

# Get the trained model
model = results['model']

# Make a prediction
diagnosis = model.diagnose(input_data)
print(model.explain_diagnosis(diagnosis, detail_level='high'))

# Visualize results
model.visualize_results(output_prefix="experiment", save_figures=True)
```

### Claude Implementation

```python
from nexus_claude import NEXUS_Claude, ClaudeLLM, EnhancedKnowledgeGraph

# Initialize Claude LLM
claude = ClaudeLLM(api_key="your-api-key", model="claude-3-opus-20240229")

# Define classes and symbols
class_names = ["Class1", "Class2"]
symbol_names = ["feature1", "feature2", "feature3"]

# Initialize knowledge graph
kg = EnhancedKnowledgeGraph()
# ... setup knowledge graph with domain knowledge ...

# Initialize NEXUS with Claude
nexus = NEXUS_Claude(
    claude_llm=claude,
    knowledge_graph=kg,
    class_names=class_names,
    symbol_names=symbol_names
)

# Make a prediction
result = nexus.diagnose("Input text describing the case")

# Get explanation
explanation = nexus.explain_diagnosis(result, detail_level='medium')
print(explanation)
```

## Architecture Overview

NEXUS consists of several key components:

1. **Neural Component**: Processes raw inputs using deep learning (either a custom transformer or Claude LLM)
2. **Symbolic Component**: Conducts logical reasoning using a knowledge graph with entities and rules
3. **Neural-Symbolic Interface**: Translates between neural representations and symbolic concepts
4. **Metacognitive Controller**: Strategically decides when to rely on each component based on confidence and risk

![Alt text](./Nexustransformer.png)

## Example Use Cases

- Medical diagnosis by combining statistical patterns with medical knowledge
- Financial risk assessment with market data and regulatory rules
- Anomaly detection with both pattern recognition and domain-specific constraints
- Decision support systems requiring both data-driven insights and explicit reasoning

## Advantages over Pure Neural or Symbolic Approaches

- Higher accuracy, especially in edge cases where one approach might fail
- Transparent decision-making with clear reasoning steps
- Reduced reliance on large training datasets through knowledge injection
- Enhanced robustness to adversarial examples and domain shifts

## Performance

The NEXUS architecture has shown promising results in comparative evaluations:

- Often outperforms both standalone neural and symbolic models
- Particularly excels in complex domains with limited training data
- Provides complementary strengths, with symbolic reasoning handling cases where neural approaches struggle

## Future Directions

- Integration with multimodal inputs (text, images, time-series)
- Extension to reinforcement learning scenarios
- Expanding the knowledge representation capabilities
- Optimization for resource-constrained environments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use NEXUS in your research, please cite our paper:

```
@article{alesso2025nexus,
  title={NEXUS-Transformer: A Neural-Symbolic Architecture for Interpretable and Aligned AI Systems},
  author={Alesso, H. P.},
  year={2025},
  url={https://www.ai-hive.net/_files/ugd/44aedb_96ebd7c4f5a14282be2e3d4613f921ce.pdf}
}
```

## Contact

- **Homepage**: [AI HIVE](https://ai-hive.net)
- **Email**: info@ai-hive.net
- **GitHub**: [https://github.com/alessoh/