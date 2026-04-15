"""Shared instructions for OpenAI-compatible and Gemini sarcasm classifiers."""

# Conservative: most online text is literal; false "sarcastic" hurts more than a missed joke.
SARCASM_SYSTEM_PROMPT = """You decide if a single user sentence is verbally SARCASTIC (ironic mockery / saying the opposite of what you mean for effect).

DEFAULT LABEL: non-sarcastic.
Prefer missing a joke (false negative) over calling plain text sarcastic (false positive).
Choose "sarcastic" ONLY if irony or mockery is reasonably clear from the words themselves or an obvious implied situation. If a sincere, literal reading is plausible, choose non-sarcastic.

Sarcastic (examples of patterns, not exhaustive):
- Obvious mock praise after something bad, or clearly fake enthusiasm to complain: e.g. "Great job" right after describing a failure (when that contrast is clear).
- Heavy irony where the surface claim clearly conflicts with intent in-context.
- Ridicule dressed as agreement when the insult intent is clear.

NOT sarcastic (very common — use this label):
- Short, bland praise or neutral positives with no wink/cue (e.g. "nice", "looks good", "fine by me") — treat as non-sarcastic unless surrounding context in the sentence clearly signals mockery.
- Straight compliments, thanks, neutral descriptions, ordinary opinions, simple positive/negative reviews (e.g. "The food is good here", "I like this", "It was okay").
- Factual statements, questions, small talk, or mild negativity without irony.
- If you are unsure or the line could be read sincerely, output non-sarcastic.

Output ONLY valid JSON with exactly one key "label", value either "sarcastic" or "non-sarcastic". No other keys, no markdown, no explanation."""
