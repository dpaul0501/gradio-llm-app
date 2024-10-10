from transformers import pipeline
from evaluate import load as load_metric

# Initialize the text generation model
generator = pipeline("text-generation", model="gpt2")

# Test 1: Checking if the generated text has hallucinations
def test_hallucination():
    prompt = "The capital of France is"
    output = generator(prompt, max_length=20, num_return_sequences=1)[0]["generated_text"]
    assert "Paris" in output, "Hallucination detected: Expected 'Paris' in the output."

# Test 2: Evaluate the LLM using BLEU score
def test_bleu_score():
    bleu = load_metric("bleu")
    reference = [["The", "capital", "of", "France", "is", "Paris"]]
    hypothesis = generator("The capital of France is", max_length=20, num_return_sequences=1)[0]["generated_text"].split()
    bleu_score = bleu.compute(predictions=[hypothesis], references=reference)
    assert bleu_score["bleu"] > 0.5, f"Low BLEU score: {bleu_score['bleu']}"

if __name__ == "__main__":
    test_hallucination()
    test_bleu_score()