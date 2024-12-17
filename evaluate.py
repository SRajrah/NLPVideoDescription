import google.generativeai as genai
from datasets import load_dataset
import pandas as pd
import json
import time
from rouge_score import rouge_scorer
import ssl
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Download the punkt tokenizer data
nltk.download('punkt')

# Initialize the genai model
genai_model = genai.GenerativeModel("gemini-1.0-pro")

# Function to generate content with backoff
def generate_with_backoff(prompt: str, max_retries: int = 5, initial_delay: float = 2.0):
    """Helper function implementing exponential backoff"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = genai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"All attempts failed for prompt after {max_retries} retries")
                return ""
            sleep_time = delay * (2 ** attempt)
            print(f"API error: {str(e)}. Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
    return ""

# Function to evaluate ROUGE scores
def evaluate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Function to evaluate BLEU scores using split for tokenization
def evaluate_bleu(reference, summary):
    reference_tokens = reference.split()
    summary_tokens = summary.split()
    smoothing_function = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], summary_tokens, smoothing_function=smoothing_function)
    return score

# Function to evaluate BERTScore
def evaluate_bertscore(reference, summary):
    P, R, F1 = score([summary], [reference], lang="en", rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

# Function to extract score from text
def extract_score(text: str) -> float:
    words = text.split()
    num = 3
    for word in words:
        try:
            num = float(word.replace('*','').replace('[','').replace(']',''))
            break
        except ValueError:
            continue
    return num

# Function to evaluate a single summary multiple times and average the scores
def evaluate_summary(summary, reference, num_evaluations=5):
    prompt_template = f"""Evaluate the following video summary based on six criteria:

    1. Descriptiveness: How well does the summary provide a rich and vivid description of the video's content? How clear is the picture it paints of what happens in the video?
    2. Coherence: How logically structured and easy to follow is the summary?
    3. Completeness: Does the summary cover all important aspects of the video?
    4. Fluency: How grammatically correct and well-written is the language in the summary?
    5. Conciseness: How well does the summary avoid unnecessary details while covering the essentials?
    6. Semantic Value: How meaningful is the video summary?

    Use the following scoring scale:
    - 1: Poor
    - 2: Below average
    - 3: Average
    - 4: Good
    - 5: Excellent

    ---
    Example Scoring:
    1. Descriptiveness

    5:
    "The summary captures detailed visuals and events from the video, such as 'the athlete’s grueling uphill training runs under the rain' or 'her heartfelt conversation with her coach.' It brings the scenes to life with vivid language, matching the richness of the video."
    4:
    "The summary describes the key moments but lacks some vivid details. For example, it mentions 'training' but doesn’t describe the intense conditions shown in the video."

    ---
    Summary: {summary}
    """

    aggregated_scores = {metric: [] for metric in ["Descriptiveness", "Coherence", "Completeness", "Fluency", "Conciseness", "Semantic Value"]}
    
    for _ in range(num_evaluations):
        response = generate_with_backoff(prompt_template)
        try:
            for metric in aggregated_scores.keys():
                if f"{metric}:" in response:
                    for line in response.split('\n'):
                        if f"{metric}" in line:
                            score_line = line
                            break
                    score = extract_score(score_line.split(f"{metric}:")[-1])
                    aggregated_scores[metric].append(score)
                else:
                    print(f"{metric} not found")
                    aggregated_scores[metric].append(1)
        except Exception as e:
            print(f"Error parsing response: {e}")
            for metric in aggregated_scores.keys():
                aggregated_scores[metric].append(0)
    
    # Calculate average scores
    average_scores = {metric: sum(scores) / len(scores) for metric, scores in aggregated_scores.items()}
    overall_average_score = sum(average_scores.values()) / len(average_scores)
    average_scores['Overall_Average'] = overall_average_score    

    # Calculate ROUGE, BLEU, and BERTScore
    rouge_scores = evaluate_rouge(reference, summary)
    bleu_score = evaluate_bleu(reference, summary)
    P, R, F1 = evaluate_bertscore(reference, summary)
    average_scores['ROUGE-1'] = rouge_scores['rouge1'].fmeasure
    average_scores['ROUGE-2'] = rouge_scores['rouge2'].fmeasure
    average_scores['ROUGE-L'] = rouge_scores['rougeL'].fmeasure
    average_scores['BLEU'] = bleu_score
    average_scores['BERTScore_P'] = P
    average_scores['BERTScore_R'] = R
    average_scores['BERTScore_F1'] = F1

    return average_scores

# Example usage
reference = "A young girl is seen sitting in a chair with a person standing next to her. The person next to her then piercing one ear followed by the other. The person rubs lotion on the piercings afterwards."
print(evaluate_summary("a close-up of a young girl with blonde hair. She appears to be around 6-7 years old and has blonde hair. She is wearing a white shirt with a pink bow on the collar. The girl is looking off to the side with a serious expression on her face. A person's hand is visible on the left side of the image, holding a blue spray bottle and applying a white substance to the girl's ear. The background is blurred, but it seems like the focus is on the girl and the person applying the substance. The girl appears to be receiving a dental procedure, as she is looking up at the dentist with a concerned expression on her face. a", reference, num_evaluations=3))