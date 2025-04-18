import asyncio
import os
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from typing import Dict, List, Tuple, Optional
import json
import time
import random
from dotenv import load_dotenv
from scipy.stats import pearsonr

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the prompts for each agent
GRAMMAR_AGENT_PROMPT = """
You are an expert English-language grammar evaluator. Your task is to analyze the grammar, mechanics, and writing quality of this student essay. 
Consider sentence structure, punctuation, spelling, verb tense consistency, word choice, and overall writing flow.

Essay to evaluate:
{essay_text}

Provide your evaluation following this exact format:
1. Grammar Analysis: [Your detailed analysis of grammar, mechanics, sentence structure]
2. Writing Quality: [Your assessment of writing flow, clarity, organization]
3. Strengths: [List key strengths related to grammar and writing]
4. Areas for Improvement: [List key areas needing improvement]
5. Score (1-6): [Your score from 1-6, where 1 is very poor and 6 is excellent]
6. Justification: [Brief explanation for your score]

Important: Your score should be based solely on grammar and writing quality, not on content or arguments.
"""

CONTENT_AGENT_PROMPT = """
You are an expert English-language content evaluator. Your task is to analyze the content, reasoning, and arguments in this student essay.
Consider thesis clarity, evidence quality, argument development, critical thinking, and overall persuasiveness.

Essay to evaluate:
{essay_text}

Provide your evaluation following this exact format:
1. Content Analysis: [Your detailed analysis of the main arguments and ideas]
2. Evidence & Reasoning: [Your assessment of how well claims are supported]
3. Strengths: [List key strengths related to content and arguments]
4. Areas for Improvement: [List key content areas needing improvement]
5. Score (1-6): [Your score from 1-6, where 1 is very poor and 6 is excellent]
6. Justification: [Brief explanation for your score]

Important: Your score should be based solely on content, arguments, and reasoning, not on grammar or writing mechanics.
"""

SUPER_AGENT_PROMPT = """
You are an expert essay evaluator. You'll review assessments from two specialized agents - one focusing on grammar and another on content.

Grammar Agent Evaluation:
{grammar_eval}

Content Agent Evaluation:
{content_eval}

Your tasks:
1. Review both evaluations
2. Determine if there is a significant discrepancy between scores (difference of more than 3 points)
3. If there is a discrepancy, explain why this might have occurred and explicitly request that both agents reevaluate the essay
4. Calculate a final score by giving EXACTLY 50% weight to grammar and 50% weight to content (average of the two scores)
5. The final score should be a whole number between 1-6 (round to nearest integer if needed)

Provide your evaluation following this exact format:
1. Grammar Assessment Review: [Your analysis of the grammar evaluation]
2. Content Assessment Review: [Your analysis of the content evaluation]
3. Discrepancy Analysis: [If scores differ by more than 2, analyze why and request reevaluation]
4. Score Calculation: Grammar score ({grammar_score}) × 0.5 + Content score ({content_score}) × 0.5 = [Show calculation]
5. Final Score (1-6): [Your calculated score, rounded to a whole number]
6. Justification: [Explanation for the final score, emphasizing equal weighting of grammar and content]

Important: 
- Grammar and content MUST be weighted equally (50% each) in your final score
- If scores differ by more than 3 points, include "SIGNIFICANT DISCREPANCY DETECTED - BOTH AGENTS SHOULD REEVALUATE" at the start of your Discrepancy Analysis section
"""

RECONSIDERATION_PROMPT = """
You previously evaluated an essay, but there was a significant discrepancy between your score and another agent's assessment. The Super Agent has requested that you carefully reevaluate the essay.

Original Essay:
{essay_text}

Your Original Evaluation:
{original_eval}

Other Agent's Evaluation:
{other_eval}

Please carefully reconsider your evaluation. Be objective and thorough. Consider whether you may have overlooked important aspects or been too harsh/lenient in your scoring.

Provide your reconsidered evaluation following the same format as before, adding a new section:
7. Reconsideration Notes: [Explain what aspects you may have initially overlooked or misinterpreted and how this affects your evaluation]

Remember to be fair and accurate. Your goal is not to match the other agent's score but to provide the most accurate assessment possible.
"""

class EssayEvaluator:
    def __init__(self, model="gpt-4o"):
        """Initialize the essay evaluator with the specified OpenAI model."""
        self.model = model

    async def evaluate_grammar(self, essay_id: str, essay_text: str) -> Dict:
        """Evaluate the grammar of an essay using the grammar agent."""
        prompt = GRAMMAR_AGENT_PROMPT.format(essay_text=essay_text)
        response = await self._get_completion(prompt)
        return self._extract_score_and_content(response, "grammar")

    async def evaluate_content(self, essay_id: str, essay_text: str) -> Dict:
        """Evaluate the content of an essay using the content agent."""
        prompt = CONTENT_AGENT_PROMPT.format(essay_text=essay_text)
        response = await self._get_completion(prompt)
        return self._extract_score_and_content(response, "content")

    async def super_agent_review(self, grammar_eval: Dict, content_eval: Dict) -> Dict:
        """Have the super agent review and reconcile the grammar and content evaluations."""
        grammar_score = grammar_eval.get("score", 0) or 0  # Default to 0 if None
        content_score = content_eval.get("score", 0) or 0  # Default to 0 if None
        
        prompt = SUPER_AGENT_PROMPT.format(
            grammar_eval=grammar_eval["full_evaluation"],
            content_eval=content_eval["full_evaluation"],
            grammar_score=grammar_score,
            content_score=content_score
        )
        response = await self._get_completion(prompt)
        return self._extract_super_agent_result(response, grammar_eval, content_eval)

    async def reconsider_evaluation(self, agent_type: str, essay_text: str, original_eval: Dict, other_eval: Dict) -> Dict:
        """Have an agent reconsider its evaluation with knowledge of the other agent's assessment."""
        prompt = RECONSIDERATION_PROMPT.format(
            essay_text=essay_text,
            original_eval=original_eval["full_evaluation"],
            other_eval=other_eval["full_evaluation"]
        )
        
        response = await self._get_completion(prompt)
        return self._extract_score_and_content(response, agent_type)

    async def _get_completion(self, prompt: str) -> str:
        """Get a completion from the OpenAI API with exponential backoff for rate limits."""
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.2,  # Low temperature for consistent scoring
                    max_tokens=2000
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

    def _extract_score_and_content(self, response: str, agent_type: str) -> Dict:
        """Extract the score and full evaluation from the agent's response."""
        try:
            # Find the score line
            lines = response.split("\n")
            score_line = ""
            for line in lines:
                if "Score (1-6):" in line:
                    score_line = line.strip()
                    break
            
            # Extract just the numeric score
            score = None
            if score_line:
                import re
                match = re.search(r"Score \(1-6\):\s*(\d+)", score_line)
                if match:
                    score = int(match.group(1))
            
            # If we couldn't find a score or it's invalid, try alternative patterns
            if score is None or score < 1 or score > 6:
                # Try looking for just a number between 1 and 6
                for line in lines:
                    match = re.search(r"(?<!\d)([1-6])(?!\d)", line)
                    if match:
                        potential_score = int(match.group(1))
                        if 1 <= potential_score <= 6:
                            score = potential_score
                            break
            
            # If still no valid score, set to None and log warning
            if score is None or score < 1 or score > 6:
                print(f"Warning: Could not extract valid score from {agent_type} agent response")
                score = None
                
            return {
                "agent_type": agent_type,
                "score": score,
                "full_evaluation": response
            }
        except Exception as e:
            print(f"Error extracting score from {agent_type} agent: {str(e)}")
            return {
                "agent_type": agent_type,
                "score": None,
                "full_evaluation": response,
                "error": str(e)
            }

    def _extract_super_agent_result(self, response: str, grammar_eval: Dict, content_eval: Dict) -> Dict:
        """Extract the final score and determine if there's a discrepancy from the super agent's response."""
        try:
            # Extract the final score
            lines = response.split("\n")
            score_line = ""
            for line in lines:
                if "Final Score (1-6):" in line:
                    score_line = line.strip()
                    break
            
            # Extract just the numeric score
            final_score = None
            if score_line:
                import re
                match = re.search(r"Final Score \(1-6\):\s*(\d+)", score_line)
                if match:
                    final_score = int(match.group(1))
            
            # If we couldn't find a score in the expected format, try alternatives
            if final_score is None or final_score < 1 or final_score > 6:
                for line in lines:
                    if "final score" in line.lower():
                        match = re.search(r"(?<!\d)([1-6])(?!\d)", line)
                        if match:
                            final_score = int(match.group(1))
                            break
            
            # Check if there's a significant discrepancy noted by the super agent
            has_discrepancy = "SIGNIFICANT DISCREPANCY DETECTED" in response
            
            # Calculate own 50/50 average as a fallback
            grammar_score = grammar_eval.get("score")
            content_score = content_eval.get("score")
            calculated_score = None
            
            if grammar_score is not None and content_score is not None:
                # If both scores are valid, calculate the 50/50 weighted average
                raw_average = (grammar_score + content_score) / 2
                calculated_score = round(raw_average)  # Round to nearest integer
                
                # Use calculated score as a fallback if no final score was found
                if final_score is None:
                    final_score = calculated_score
                    print("Using calculated 50/50 average as final score")
                
                # Also check direct score difference
                score_diff = abs(grammar_score - content_score)
                if score_diff > 2:
                    has_discrepancy = True
            
            return {
                "agent_type": "super",
                "final_score": final_score,
                "calculated_score": calculated_score,  # Include the calculated score
                "grammar_score": grammar_score,
                "content_score": content_score,
                "has_discrepancy": has_discrepancy,
                "full_evaluation": response
            }
        except Exception as e:
            print(f"Error extracting super agent result: {str(e)}")
            return {
                "agent_type": "super",
                "final_score": None,
                "grammar_score": grammar_eval.get("score"),
                "content_score": content_eval.get("score"),
                "has_discrepancy": False,
                "full_evaluation": response,
                "error": str(e)
            }

    async def evaluate_essay(self, essay_id: str, essay_text: str) -> Dict:
        """
        Evaluate an essay using all agents with potential reconsideration based on discrepancy.
        Ensures 50/50 weighting between grammar and content scores.
        """
        print(f"Evaluating essay {essay_id}...")
        
        # Run grammar and content evaluations concurrently
        grammar_task = asyncio.create_task(self.evaluate_grammar(essay_id, essay_text))
        content_task = asyncio.create_task(self.evaluate_content(essay_id, essay_text))
        
        grammar_eval = await grammar_task
        content_eval = await content_task
        
        # Print initial scores
        print(f"Initial scores - Grammar: {grammar_eval['score']}, Content: {content_eval['score']}")
        
        # Have the super agent review the evaluations with equal weighting
        super_eval = await self.super_agent_review(grammar_eval, content_eval)
        
        # If there's a significant discrepancy, have the agents reconsider
        if super_eval["has_discrepancy"]:
            print(f"Significant discrepancy detected for essay {essay_id}. Requesting reevaluation...")
            
            # Run reconsiderations concurrently with knowledge of each other's evaluations
            new_grammar_task = asyncio.create_task(
                self.reconsider_evaluation("grammar", essay_text, grammar_eval, content_eval)
            )
            new_content_task = asyncio.create_task(
                self.reconsider_evaluation("content", essay_text, content_eval, grammar_eval)
            )
            
            new_grammar_eval = await new_grammar_task
            new_content_eval = await new_content_task
            
            # Print reconsidered scores
            print(f"Reconsidered scores - Grammar: {new_grammar_eval['score']}, Content: {new_content_eval['score']}")
            
            # Have the super agent review the new evaluations
            new_super_eval = await self.super_agent_review(new_grammar_eval, new_content_eval)
            
            # Store all evaluations including the reconsidered ones
            return {
                "essay_id": essay_id,
                "initial_grammar_eval": grammar_eval,
                "initial_content_eval": content_eval,
                "initial_super_eval": super_eval,
                "reconsidered_grammar_eval": new_grammar_eval,
                "reconsidered_content_eval": new_content_eval,
                "final_super_eval": new_super_eval,
                "final_score": new_super_eval["final_score"],
                "required_reconsideration": True
            }
        else:
            # No significant discrepancy, ensure the score is a 50/50 blend
            # (Either from super_eval or calculated manually if needed)
            if super_eval.get("calculated_score") and super_eval.get("final_score") != super_eval.get("calculated_score"):
                print(f"Note: Super agent score {super_eval['final_score']} differs from 50/50 average {super_eval['calculated_score']}")
                print(f"Using 50/50 average score as specified")
                final_score = super_eval.get("calculated_score")
            else:
                final_score = super_eval["final_score"]
                
            return {
                "essay_id": essay_id,
                "grammar_eval": grammar_eval,
                "content_eval": content_eval,
                "super_eval": super_eval,
                "final_score": final_score,
                "required_reconsideration": False
            }

async def process_essay_batch(batch: List[Dict], evaluator: EssayEvaluator) -> List[Dict]:
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
    evaluator = EssayEvaluator()
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
            
            # Print a summary of the result
            if result.get("required_reconsideration", False):
                print(f"Essay {result['essay_id']}: Initial G:{result['initial_grammar_eval']['score']} C:{result['initial_content_eval']['score']} → " 
                      f"Reconsidered G:{result['reconsidered_grammar_eval']['score']} C:{result['reconsidered_content_eval']['score']} → "
                      f"Final:{result['final_score']} (50/50 weight)")
            else:
                grammar_score = result['grammar_eval']['score']
                content_score = result['content_eval']['score']
                expected_avg = round((grammar_score + content_score) / 2) if grammar_score and content_score else None
                print(f"Essay {result['essay_id']}: G:{grammar_score} C:{content_score} → Final:{result['final_score']} (50/50 weight = {expected_avg})")
        
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
        print("Starting essay evaluation system with 50/50 grammar/content weighting...")
        
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
        reconsideration_count = sum(1 for r in results.values() if r.get("required_reconsideration", False))
        print(f"\n======= Evaluation Complete =======")
        print(f"Total essays processed: {len(results)}")
        print(f"Essays requiring reconsideration: {reconsideration_count} ({reconsideration_count/len(results)*100:.1f}%)")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average time per essay: {elapsed_time/len(results):.1f} seconds")
        
        # Verify 50/50 weighting
        print("\n======= 50/50 Weighting Verification =======")
        all_correct = True
        for essay_id, result in results.items():
            if result.get("required_reconsideration", False):
                grammar_score = result['reconsidered_grammar_eval']['score']
                content_score = result['reconsidered_content_eval']['score']
            else:
                grammar_score = result['grammar_eval']['score']
                content_score = result['content_eval']['score']
                
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
            if score is None:
                if "final_super_eval" in result:
                    score = result.get("final_super_eval", {}).get("final_score")
                else:
                    score = result.get("super_eval", {}).get("final_score")
            
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
                "essays_requiring_reconsideration": reconsideration_count,
                "reconsideration_percentage": reconsideration_count/len(results)*100,
                "weighting": "50% grammar, 50% content"
            }
        else:
            evaluation_summary = {
                "metrics": "No true scores available for evaluation",
                "processing_time": elapsed_time,
                "essays_processed": len(results),
                "essays_requiring_reconsideration": reconsideration_count,
                "reconsideration_percentage": reconsideration_count/len(results)*100,
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