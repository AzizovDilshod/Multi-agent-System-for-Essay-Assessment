import asyncio
import os
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from typing import Dict, List
import json
import time
import random
from dotenv import load_dotenv
from scipy.stats import pearsonr

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a comprehensive prompt that combines everything in one request
EVALUATOR_PROMPT = """
You are an expert essay evaluator. Analyze both grammar/mechanics and content/arguments for this essay:

{essay_text}

Evaluate using this format:
1. Grammar Analysis: [Analyze grammar, mechanics, structure]
2. Content Analysis: [Analyze arguments and ideas]
3. Writing Quality: [Assess flow, clarity, organization]
4. Evidence & Reasoning: [Assess how claims are supported]
5. Strengths: [Key strengths]
6. Areas for Improvement: [Areas needing improvement]
7. Grammar Score (1-6): [Score from 1-6]
8. Content Score (1-6): [Score from 1-6]
9. Final Score (1-6): [Calculate as (Grammar Score + Content Score) ÷ 2, rounded]
10. Justification: [Brief explanation]

Keep grammar score focused on mechanics only and content score on arguments only. Final score must be 50/50 weighted average.
"""

class SingleLLMEssayEvaluator:
    def __init__(self, model="gpt-4"):
        """Initialize the essay evaluator with the OpenAI model."""
        # Use a standard model name - can be changed based on what's accessible
        self.model = model

    async def evaluate_essay(self, essay_id: str, essay_text: str) -> Dict:
        """Evaluate an essay using a single LLM that handles both grammar and content."""
        print(f"Evaluating essay {essay_id}...")
        
        prompt = EVALUATOR_PROMPT.format(essay_text=essay_text)
        response = await self._get_completion(prompt)
        result = self._extract_scores_and_content(response, essay_id)
        
        print(f"Essay {essay_id}: G:{result['grammar_score']} C:{result['content_score']} → Final:{result['final_score']}")
        
        return result

    async def _get_completion(self, prompt: str) -> str:
        """Get a completion from the OpenAI API with exponential backoff for rate limits."""
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Using standard parameters for closed-source model
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    # No custom parameters that might not be available
                )
                return response.choices[0].message.content
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                elif attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"API error: {str(e)}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    raise

    def _extract_scores_and_content(self, response: str, essay_id: str) -> Dict:
        """Extract all scores and the full evaluation from the response."""
        try:
            # Find the score lines
            lines = response.split("\n")
            grammar_score_line = ""
            content_score_line = ""
            final_score_line = ""
            
            for line in lines:
                if "Grammar Score (1-6):" in line:
                    grammar_score_line = line.strip()
                elif "Content Score (1-6):" in line:
                    content_score_line = line.strip()
                elif "Final Score (1-6):" in line:
                    final_score_line = line.strip()
            
            # Extract the numeric scores using regex
            import re
            grammar_score = None
            content_score = None
            final_score = None
            
            if grammar_score_line:
                match = re.search(r"Grammar Score \(1-6\):\s*(\d+)", grammar_score_line)
                if match:
                    grammar_score = int(match.group(1))
            
            if content_score_line:
                match = re.search(r"Content Score \(1-6\):\s*(\d+)", content_score_line)
                if match:
                    content_score = int(match.group(1))
            
            if final_score_line:
                match = re.search(r"Final Score \(1-6\):\s*(\d+)", final_score_line)
                if match:
                    final_score = int(match.group(1))
            
            # Calculate expected final score (if grammar and content scores are available)
            expected_final_score = None
            if grammar_score is not None and content_score is not None:
                expected_final_score = round((grammar_score + content_score) / 2)
                
                # Use expected score if no final score was found
                if final_score is None:
                    final_score = expected_final_score
                    print(f"No final score found, using calculated average: {final_score}")
                # Check if final score matches expected 50/50 weighting
                elif final_score != expected_final_score:
                    print(f"Warning: Final score {final_score} doesn't match expected 50/50 average {expected_final_score}")
                    final_score = expected_final_score
            
            return {
                "essay_id": essay_id,
                "grammar_score": grammar_score,
                "content_score": content_score,
                "final_score": final_score,
                "full_evaluation": response
            }
        except Exception as e:
            print(f"Error extracting scores: {str(e)}")
            return {
                "essay_id": essay_id,
                "grammar_score": None,
                "content_score": None,
                "final_score": None,
                "full_evaluation": response,
                "error": str(e)
            }

async def process_essay_batch(batch: List[Dict], evaluator: SingleLLMEssayEvaluator) -> List[Dict]:
    """Process a batch of essays concurrently."""
    tasks = []
    for essay in batch:
        # Create a task for each essay
        task = asyncio.create_task(
            evaluator.evaluate_essay(
                essay["essay_id"], 
                essay["full_text"]
            )
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    return await asyncio.gather(*tasks)

async def evaluate_essays_dataset(essays_data: List[Dict], batch_size: int = 5) -> Dict[str, Dict]:
    """
    Process an entire dataset of essays with proper batching and rate limiting.
    
    Args:
        essays_data: List of dictionaries with essay_id and full_text
        batch_size: Number of essays to evaluate concurrently
        
    Returns:
        Dictionary mapping essay_id to evaluation results
    """
    evaluator = SingleLLMEssayEvaluator()
    all_results = {}
    
    # Process essays in batches
    for i in range(0, len(essays_data), batch_size):
        batch = essays_data[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(essays_data)-1)//batch_size + 1}, essays {i+1}-{min(i+batch_size, len(essays_data))}")
        
        # Process the batch
        batch_results = await process_essay_batch(batch, evaluator)
        
        # Add results to our collection
        for result in batch_results:
            all_results[result["essay_id"]] = result
        
        # Add a delay between batches to avoid rate limits
        if i + batch_size < len(essays_data):
            delay = 2  # seconds
            print(f"Waiting {delay}s before next batch...")
            await asyncio.sleep(delay)
    
    return all_results

def calculate_evaluation_metrics(predictions, actual_scores):
    """
    Calculate evaluation metrics to assess model performance.
    
    Args:
        predictions: List of predicted scores
        actual_scores: List of actual (true) scores
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Filter out any None values
    valid_pairs = [(p, a) for p, a in zip(predictions, actual_scores) if p is not None]
    if not valid_pairs:
        return {
            "standard_deviation": None,
            "mae": None,
            "pearson_correlation": None,
            "valid_predictions": 0,
            "total_predictions": len(predictions)
        }
    
    pred_scores, true_scores = zip(*valid_pairs)
    
    # Convert to numpy arrays for calculations
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    
    # Calculate standard deviation of errors
    errors = pred_scores - true_scores
    std_dev = np.std(errors)
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(errors))
    
    # Calculate Pearson correlation coefficient
    pearson_corr, p_value = pearsonr(pred_scores, true_scores)
    
    return {
        "standard_deviation": std_dev,
        "mae": mae,
        "pearson_correlation": pearson_corr,
        "p_value": p_value,
        "valid_predictions": len(valid_pairs),
        "total_predictions": len(predictions)
    }

async def main():
    try:
        print("Starting single LLM essay evaluation system with 50/50 grammar/content weighting...")
        
        # Load essays data
        try:
            # Attempt to load from CSV
            print("Loading essays data from CSV...")
            df = pd.read_csv('training_essays.csv')
            
            # Check if 'score' column exists for evaluation metrics
            has_true_scores = 'score' in df.columns
            if has_true_scores:
                print("True scores found in CSV. Will calculate evaluation metrics.")
            else:
                print("No 'score' column found. Will generate predictions only.")
                
            essays_data = df[['essay_id', 'full_text']].to_dict('records')
            if has_true_scores:
                # Add true scores to each essay record
                for i, essay in enumerate(essays_data):
                    essay['true_score'] = df.loc[i, 'score']
                    
            print(f"Loaded {len(essays_data)} essays from CSV.")
        except FileNotFoundError:
            # If CSV not found, use sample data for testing
            print("CSV file not found. Using sample data for demonstration.")
            essays_data = [
                {
                    "essay_id": "sample_1",
                    "full_text": "This is a sample essay with good grammar but weak content. The sentences are well-structured and there are no major errors. However, the arguments lack depth and evidence.",
                    "true_score": 4
                },
                {
                    "essay_id": "sample_2",
                    "full_text": "This sample essay has poor grammar with many errors. However the content is thoughtful with strong arguments supported by evidence and critical thinking even though the writing mechanics are weak.",
                    "true_score": 3
                },
                {
                    "essay_id": "sample_3",
                    "full_text": "This is an average essay with both average grammar and average content. It makes some good points but also has some structural issues and grammatical errors.",
                    "true_score": 3
                }
            ]
            has_true_scores = True
        
        # For testing, limit to a few essays
        # Comment this out to process the entire dataset
        if len(essays_data) > 5:
            print("Using only the first 5 essays for this demonstration run.")
            essays_data = essays_data[:5]
        
        # Start timing
        start_time = time.time()
        
        # Evaluate essays
        results = await evaluate_essays_dataset(essays_data, batch_size=2)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Report statistics
        print(f"\n======= Evaluation Complete =======")
        print(f"Total essays processed: {len(results)}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average time per essay: {elapsed_time/len(results):.1f} seconds")
        
        # Verify 50/50 weighting
        print("\n======= 50/50 Weighting Verification =======")
        all_correct = True
        for essay_id, result in results.items():
            grammar_score = result['grammar_score']
            content_score = result['content_score']
                
            if grammar_score is not None and content_score is not None:
                expected_score = round((grammar_score + content_score) / 2)
                actual_score = result['final_score']
                
                if expected_score != actual_score:
                    print(f"Warning: Essay {essay_id} has unexpected score. Expected {expected_score}, got {actual_score}")
                    all_correct = False
        
        if all_correct:
            print("All essays correctly received 50/50 weighting between grammar and content scores.")
        
        # Create predictions dataframe
        predictions = []
        for essay_id, result in results.items():
            score = result.get("final_score")
            
            # Find true score if available
            true_score = None
            if has_true_scores:
                for essay in essays_data:
                    if essay['essay_id'] == essay_id:
                        true_score = essay.get('true_score')
                        break
            
            predictions.append({
                "essay_id": essay_id,
                "predicted_score": score,
                "true_score": true_score
            })
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate evaluation metrics if true scores are available
        if has_true_scores:
            # Extract predictions and true scores
            pred_scores = predictions_df['predicted_score'].tolist()
            true_scores = predictions_df['true_score'].tolist()
            
            # Calculate metrics
            metrics = calculate_evaluation_metrics(pred_scores, true_scores)
            
            # Print evaluation metrics
            print("\n======= Evaluation Metrics =======")
            print(f"Standard Deviation of Errors: {metrics['standard_deviation']:.4f}")
            print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
            print(f"Pearson Correlation Coefficient: {metrics['pearson_correlation']:.4f} (p-value: {metrics['p_value']:.4f})")
            print(f"Valid predictions: {metrics['valid_predictions']}/{metrics['total_predictions']}")
            
            # Add metrics to results JSON
            evaluation_summary = {
                "metrics": metrics,
                "processing_time": elapsed_time,
                "essays_processed": len(results),
                "weighting": "50% grammar, 50% content"
            }
        else:
            evaluation_summary = {
                "metrics": "No true scores available for evaluation",
                "processing_time": elapsed_time,
                "essays_processed": len(results),
                "weighting": "50% grammar, 50% content"
            }
        
        # Save predictions to CSV
        predictions_df.to_csv('essay_predictions.csv', index=False)
        print(f"Predictions saved to essay_predictions.csv")
        
        # Save detailed results to JSON
        with open('essay_evaluations.json', 'w') as f:
            # Create a combined results dictionary with both evaluations and summary
            combined_results = {
                "evaluations": results,
                "summary": evaluation_summary
            }
            json.dump(combined_results, f, indent=2, default=str)
        print(f"Detailed evaluations saved to essay_evaluations.json")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
