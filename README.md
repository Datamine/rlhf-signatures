# rlhf-signatures

### How to use

This project uses the UV python package manager. Install UV locally, then run inside this directory:
```
uv venv
source .venv/bin/activate
uv sync
```
And you're good to go.

1. Optional: run `generate_questions.py` to generate questions.csv. You need this only if you want to modify the questions.
2. Run `get_answers.py`. This is resistant to errors, and will skip over previously answered questions.
3. Run `merge_spreadsheets.py` to combine the answers back with the questions. You need this only because of an oversight on my part. (See TODO)
4. Run `validate_answers.py dataset/` to check all the answers for validity. There's a small amount of cleanup that may be necessary.

### TODO
1. Fix `get_answers.py` to replicate the question text when writing answers, so we don't need `merge_spreadsheets.py`.
2. Find a good non-RLHF benchmark model. I tried EleutherAI/GPT-NeoX-20B but it was terrible. I couldn't find a hosted version of Bloom-176B.
