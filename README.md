# RLHF Signatures

### Getting Started

This project uses the UV Python package manager. [Install UV locally](https://docs.astral.sh/uv/getting-started/installation/),
then run inside this directory:

```
uv venv
source .venv/bin/activate
uv sync
```

That will install all the required packages to this virtual environment, and you'll be able to run all the scripts herein.
This project is configured for Python 3.12, so you may need to fiddle with the configuration if your system has a lower Python version.
(Practically, it shouldn't make a difference.)

### Running Your Own Experiments

For replicating my results, you can follow these steps:

1. Ensure that you have the following environment variables set:
```
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export TOGETHER_AI_API_KEY="..."
```

2. Optional: run `generate_questions.py` to generate questions.csv. This will generate the same questions as in the `questions/` directory.
You can modify the wording inside `generate_questions.py` to generate your own question sets. I recommend truncating your question set into a `questions_short.csv` version
for testing. One thing that's **very important** is that the questions here are designed to be symmetric:
for every question that asks the LLM to choose between Option A and Option B, there's an identical question that asks the LLM to choose between Option B and Option A.
As the later sections show, some models have significant ordering biases, so creating question sets that are perfectly symmetrical with respect to ordering is
important to counteract any biasing effects.

3. Run `get_answers.py`. This will query all the APIs with the questions from `questions.csv`. It is designed to be robust errors/interruptions, and therefore handles
every single row as a separate file-write. It will skip over all previously written questions. Pay attention to the logs in case of rate limiting, as I encountered with Gemini 2.0 Pro.

4. Run `validate_answers.py dataset/` to check all the answers for validity. There's a small amount of cleanup that may be necessary: Gemini's models terminate all their answers with a newline, and various models will insert periods, and occasionally answer with a whole sentence.
This takes only a few minutes to review and clean up in Excel or Vim -- I recommend cleaning up the data yourself, as it's a good chance to look closely at it, run a sanity chec, and notice any patterns.

5. Run `naive_order_bias.py` on each of your answer files to test whether the LLM is systematically biased toward answering with the first (or second) option.
It outputs a Binomial Test and a Chi-Square Test against the Null Hypothesis that there is no ordering bias.

6. Run `paired_order_bias.py` on each of your answer files as a third ordering test: it runs a paired t-test to evaluate whether the mean difference
in proportions (between the two orderings, Option A vs Option B and Option B vs Option A) is statistically different from zero. The t-statistic indicates
whether or not the ordering has an effect on the choice. (What's different about this test is that it explicitly uses the symmetrical ordering pairs, whereas the Binomial and Chi-Square tests don't.)

7. Run `bradley_terry.py` to compute the Bradley-Terry parameters to evaluate the relative strength of preference for each
of the Options within a given model run. You can run it on an individual file (`python bradley_terry.py answers/...`) to get the full
statistics printout, or run it on a whole directory (`python bradley_terry.py answers/`) to output a spreadsheet of all the strength parameters,
i.e. as a table over all the Options and the Models.

### TODO

1. Find a good non-RLHF benchmark model.
I couldn't find a good non-RLHF benchmark model.
I tried [EleutherAI/GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b), but it was too bad to to be useful.
I couldn't find a hosted version of [Bloom-176B](https://huggingface.co/bigscience/bloom).

2. Combine `naive_order_bias.py` and `paired_order_bias.py` into one file and  have them output to a spreadsheet, rather than printing the results.

3. Write an explanation of the results section.

### FAQ

**Q: How do I contact you with my question?**

A: Write to contact@johnloeber.com

**Q: How much did it cost to run these experiments?**

A: Maybe $20 in total. My calls to Llama-405B cost me only $0.10 total via TogetherAI. DeepSeek cost me ~50 cents, even with their reasoning model.
OpenAI managed to charge me $15 for 200,000 output tokens, which had me mystified until I remembered that they don't just charge for output, but
for the Chain-of-Thought tokens, since O1 is a reasoning model.

**Q: How can I contribute?**

A: Feel free to file issues, PRs, etc. I have GitHub notifications turned off, so email me if there's something I should look at.
