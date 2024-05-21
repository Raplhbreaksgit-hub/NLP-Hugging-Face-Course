from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")

classifier1 = pipeline("zero-shot-classification")
result1 = classifier1("I have got JEE advanced rank of 5000 and would now take admission in IIT Roorkee with Metallurgy branch . I am excited",candidate_labels=["education", "infrastructure", "business","emotions"])

# Similarly code for generator 

# Now using the model of our choice from the HUB.
# Initialize the text generation pipeline with distilgpt2 model
generator = pipeline("text-generation", model="distilgpt2")

# Generate text with specified parameters
result = generator(
    "I am Aman Agrawal, I have cleared the IIT exam with a rank of 6851 and took materials engg. branch and further I",
    max_new_tokens=30,  # Specify the number of tokens to generate
    num_return_sequences=2  # Number of sequences to return
)
# Print the generated results
for idx, sequence in enumerate(result):
    print(f"Result {idx + 1}: {sequence['generated_text']}")

# Masking
unmasker = pipeline("fill-mask",model = "google-bert/bert-base-multilingual-cased")
results = unmasker("This course will teach you all about [MASK] models.", top_k=2)
# Print the results including token ID, token string, and score
for result in results:
    print(f"Token ID: {result['token']}, Token: {result['token_str']}, Score: {result['score']:.4f}")

