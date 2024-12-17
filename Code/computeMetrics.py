import json
import pandas as pd

def compute_metrics_with_adjusted_ranks(results_file, excel_file):
    # Load the query-answer pairs from the Excel file
    df = pd.read_excel(excel_file, sheet_name=0)
    query_to_answer = dict(zip(df["sentence_A"], df["sentence_B"]))

    with open(results_file, 'r') as f:
        data = json.load(f)

    total_queries = len(data)
    reciprocal_ranks = []
    correct_at_adjusted_rank_1 = 0  # Tracking answers logically at rank 1 (treating rank 2 as 1)
    similarity_scores = []

    for item in data:
        query_sentence = item["querySentence"]["sentence"]
        matching_sentences = item["matchingSentences"]

        # Get the correct answer for the current query
        correct_answer = query_to_answer.get(query_sentence)
        if not correct_answer:
            print(f"Warning: No matching answer found for query: {query_sentence}")
            continue
        
        
        # Track the rank of the correct answer
        correct_rank = None
        for match in matching_sentences:
            if match['rank'] == 2:
                similarity_scores.append(match['sim_score'])
            if match["sentence"] == correct_answer:
                correct_rank = match["rank"]
                break


        # Adjust the rank logic: Treat rank 2 as rank 1
        if correct_rank == 2:
            correct_at_adjusted_rank_1 += 1
            reciprocal_ranks.append(1.0)  # Treat as rank 1
        elif correct_rank is not None:  # If found but not at rank 2
            reciprocal_ranks.append(1 / (correct_rank - 1))
            print("query - correct answer  = ",query_sentence, " | ", correct_answer, correct_rank)

        else:  # If not found in the top 5
            reciprocal_ranks.append(0)

    # Calculate evaluation metrics
    mrr = sum(reciprocal_ranks) / total_queries
    precision_at_1 = correct_at_adjusted_rank_1 / total_queries
    mean_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0


    # Print the metrics
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Adjusted Precision@1 (P@1): {precision_at_1:.4f}")
    print(f"Mean Similarity Score: {mean_similarity_score:.4f}")

# Compute metrics
# File containing the JSON results
results_file = "../Dataset/semantic_results.json"

# Excel file with query-answer pairs
excel_file = "../Dataset/SentencePairsSheet.xlsx"

# Compute metrics
compute_metrics_with_adjusted_ranks(results_file, excel_file)