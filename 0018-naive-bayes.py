from collections import defaultdict
import math

def train_naive_bayes(spam_counts, not_spam_counts, vocab):
    total_spam_words = sum(spam_counts.values())
    total_not_spam_words = sum(not_spam_counts.values())
    spam_probs = {}
    not_spam_probs = {}
    
    for word in vocab:
        spam_probs[word] = (spam_counts[word] + 1) / (total_spam_words + len(vocab))
        not_spam_probs[word] = (not_spam_counts[word] + 1) / (total_not_spam_words + len(vocab))
    
    return spam_probs, not_spam_probs, total_spam_words, total_not_spam_words

def classify_message(message, spam_probs, not_spam_probs, total_spam, total_not_spam, vocab):
    log_spam_prob = 0
    log_not_spam_prob = 0
    
    for word in message.split():
        if word in vocab:
            log_spam_prob += math.log(spam_probs[word])
            log_not_spam_prob += math.log(not_spam_probs[word])
    return "Spam" if log_spam_prob > log_not_spam_prob else "Not Spam"

spam_counts = defaultdict(int, {"You": 2, "won": 20, "a": 0, "lakh": 30})
not_spam_counts = defaultdict(int, {"You": 12, "won": 1, "a": 100, "lakh": 0, "I": 50})
vocab = set(spam_counts.keys()).union(set(not_spam_counts.keys()))
spam_probs, not_spam_probs, total_spam, total_not_spam = train_naive_bayes(spam_counts, not_spam_counts, vocab)
message = "You won a lakh"
result = classify_message(message, spam_probs, not_spam_probs, total_spam, total_not_spam, vocab)

print(f"Message: '{message}' is classified as: {result}")
