# Multi-Agent Essay Scoring System

This project implements a sophisticated multi-agent system for scoring student essays using GPT-4o. It utilizes specialized agents for grammar and content evaluation, with a super agent that synthesizes the assessments to produce a final score.

## Key Features

- **Multi-Agent Architecture**: Separates grammar and content evaluation for specialized assessment
- **Equal Weighting**: Combines grammar and content scores with exact 50/50 weighting
- **Automatic Reconsideration**: Detects score discrepancies and triggers reevaluation
- **Comprehensive Evaluation**: Generates detailed feedback along with numeric scores
- **Performance evaluation measures**: Calculates MAE, Standard Deviation, and Pearson Correlation

## System Architecture

The system consists of three specialized agents:

1. **Grammar Agent**: Evaluates grammar, mechanics, sentence structure, and writing quality
2. **Content Agent**: Assesses arguments, evidence, reasoning, and overall persuasiveness
3. **Super Agent**: Reviews both evaluations, ensures equal weighting, and handles discrepancies

When a significant discrepancy (>3 points) occurs between grammar and content scores, the system automatically triggers a reconsideration process where both agents review each other's evaluations for more balanced assessment.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key with access to GPT-4o (can be substitute by another option)

### Installation

1. Clone this repository
   ```bash
   git clone https://github.com/
   cd essay-scoring-system
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## Usage

### Basic Usage

1. Prepare your CSV file with at least `essay_id` and `full_text` columns. For evaluation evaluation measures, include a `score` column with true scores.

2. Run the main script
   ```bash
   python essay_scoring_system.py
   ```

3. Check the results in:
   - `essay_predictions.csv`: CSV file with predicted scores
   - `essay_evaluations.json`: Detailed JSON with full evaluations and evaluation measures

### Configuration Options

You can adjust the following parameters in the script:

- `model`: GPT model to use (default: "gpt-4o")
- Evaluation prompts in the script

## Evaluation evaluation measures

The system calculates three key evaluation measures to evaluate performance:

1. **Mean Absolute Error (MAE)**: Average difference between predicted and true scores
2. **Standard Deviation of Errors**: Consistency of predictions
3. **Pearson Correlation Coefficient**: Linear relationship between predicted and true scores

## Sample Data

The repository includes sample essays for testing. To use your own data, create a CSV file with the following structure:

```
essay_id,full_text,score
1,"This is the full text of essay 1...",5
2,"This is the full text of essay 2...",3
...
```

## Results Interpretation

The system provides detailed evaluations for each essay, including:

- Grammar and content analysis
- Strengths and areas for improvement
- Score justification
- Final 50/50 weighted score


## Acknowledgments

- This system was designed to fairly evaluate student essays using a multi-agent approach
- We use openly avaiable dataset provided in kaggle competition, which is cited in our manuscript
- Inspired by educational rubrics for holistic essay assessment