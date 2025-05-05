import argparse
from transformers import pipeline


def load_texts(input_file):
    with open(input_file, 'r') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    return texts


def classify_texts(texts, candidate_labels):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = []
    for text in texts:
        result = classifier(text, candidate_labels=candidate_labels)
        results.append({
            "text": text,
            "labels": result['labels'],
            "scores": result['scores']
        })
    return results


def print_results(results):
    for result in results:
        print(f"\nText: {result['text']}")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  - {label}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Few-shot classification using BART")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--labels', nargs='+', default=["benign", "malignant"], help='Candidate labels')
    args = parser.parse_args()

    texts = load_texts(args.input_file)
    results = classify_texts(texts, args.labels)
    print_results(results)


if __name__ == '__main__':
    main()
