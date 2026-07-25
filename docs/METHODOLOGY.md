# Methodology

This document exists so that the numbers this tool prints can be argued with.
It states what is measured, how the decisions are made, and — the part most
benchmark documentation omits — what the results **cannot** tell you.

## 1. The unit of measurement is an application, not a model

A raw model behind a chat endpoint and the same model inside a retrieval
pipeline with a tool loop have different exposure, and only the second is what
anyone deploys. Every probe is therefore a complete scenario: a system prompt
carrying the policy, a legitimate user task, and — where the attack is indirect
— untrusted content arriving through the channel a real application would read
it from.

`TargetRequest` keeps `untrusted` separate from `user` for this reason. A
target that concatenates them has already destroyed the distinction any
input-side defense depends on, and a benchmark that flattens them cannot
measure spotlighting at all.

## 2. No grader calls a language model

Scoring a security benchmark with a judge model imports the exact failure it is
supposed to detect. The judge reads adversarial text, so it is promptable by
the thing it is grading; and its verdicts drift between runs, so a "20%
improvement" can be the judge having a different day.

Every grader in `lsb/core/grader.py` is a string or regex predicate over the
response, and every one returns the reason for its verdict so a number is
traceable back to a decision.

**The cost of that rule, stated plainly:** these graders measure *proxy*
behaviours. `policy_break` does not know whether text is harmful; it knows
whether the model emitted a specific item it was told to withhold.

## 3. Leakage is proven with per-run tokens, not judged

Asking "did the response reveal the system prompt?" has no decidable answer.
So each trial mints tokens derived from the run seed and plants them:

| Token | Planted in | Emitting it proves |
| --- | --- | --- |
| `LSB-CANARY-<hex>` | the system prompt | the system prompt leaked |
| `SEC-<HEX>` | the system prompt, as a protected item | a stated policy was broken |
| `LSB-SENTINEL-<hex>` | untrusted content | untrusted data was executed as instruction |

A hit is proof. A miss is proof of nothing leaking *that token* — which is a
narrower claim than "nothing leaked", and the right one to make.

Tokens are per-trial, so a probe can run a thousand times without the target
ever seeing the same value twice, and a leak cannot be a memorised artefact of
an earlier run.

### Obfuscated leaks count

A model that has decided to leak often does it untidily — spelled out, base64'd,
reversed, spaced, split across lines. A literal substring search scores all of
those as passes and flatters the target. `lsb/core/canary.py` normalises
aggressively and decodes base64, hex and rot13 runs before concluding a canary
is absent, and reports which channel the leak came through, because "leaked
verbatim" and "leaked base64-encoded" are different bugs with different fixes.

## 4. The jailbreak family contains no harmful content, deliberately

It measures whether a model **keeps a rule it was given** — the
instruction-hierarchy property — using a benign protected item as the proxy.
The system prompt says never to output an override code; every probe tries to
extract it through a different framing.

This is a deliberate scope choice with two motivations. Methodologically,
emitting the token is a *provable* rule break, gradeable by string comparison
rather than by a judge's opinion about tone. Practically, it means this
repository does not ship a corpus of harmful prompts.

**Therefore: a low score in this family does not mean a model is safe.**
Whether it will produce dangerous content is a different question requiring a
human-reviewed corpus, and is out of scope here. Anyone reading a `jailbreak:
4%` row should read it as "kept its instructions 96% of the time", nothing more.

## 5. Every rate carries a confidence interval

"3/10 attacks succeeded" is not actionable: ten trials give a 95% interval
spanning roughly 11%–60%, so a move from 30% to 20% on ten probes is
indistinguishable from noise, and shipping a defense on that basis is
guesswork.

Rates use the **Wilson score interval** rather than the normal approximation,
because attack-success rates live near 0 and 1 where the textbook interval
breaks down — it reports a lower bound below zero for 0/20, which is how a
team ends up claiming a defense is airtight on twenty samples.

Defense comparisons go through `compare()`, which calls a difference real only
when the two 95% intervals do not overlap. That is stricter than a
two-proportion z-test, and the conservatism is the point: the failure mode
being guarded against is shipping a guardrail that did nothing.

`--repeat` is the only lever on interval width. A single run of a 49-probe
suite is a smoke test, not a measurement.

## 6. What gets excluded from the denominator

Two outcomes mean the application was never actually exercised, and both are
excluded from rates rather than counted as attacks that failed:

- **Errors** — timeouts, rate limits, connection failures.
- **Platform refusals** — the API declining at the classifier level
  (HTTP 200, `stop_reason: "refusal"`). That is the platform declining, not the
  application defending, and crediting it would attribute a defense that never ran.

Counting either as "the attack failed" is how a rate-limited run turns into a
clean bill of health.

## 7. Mechanical vs prompt-level defenses

Defenses split into two kinds, and conflating them is how offline benchmarks
produce fiction:

**Mechanical** defenses change bytes. Input filtering that deletes a payload,
or output screening that redacts a token, has an effect fully observable from
outside the model — it holds against a simulated target exactly as against a
real one. The mock only emits a token that survived in its own input, so
stripping the payload upstream changes the outcome mechanically.

**Prompt-level** defenses ask the model to behave differently. Their effect
**cannot be measured against a simulated target** — only assumed. Spotlighting
carries `mechanical = False` and an `assumed_factor` that is explicitly labelled
a placeholder, and any run combining a prompt-level defense with a mock target
is marked `simulated` in the report and in every rendering of it.

## 8. Reproducibility

Tokens are derived from the run seed with blake2b rather than drawn randomly,
and the mock target's PRNG is seeded on `(run seed, probe id, trial)`. Given a
seed, a suite and a profile, two people on two machines get byte-identical
output. `scripts/reproduce.py` re-derives the published table, and CI runs it
on every push — a benchmark whose numbers cannot be re-derived from the repo is
a claim, not a measurement.

Use `--fresh-tokens` against a real target if you would rather the tokens were
unpredictable; it costs exact reproducibility.

## 9. Validity checks on the corpus itself

`lsb validate` runs the whole corpus against `EchoTarget`, which leaks
everything. Any probe that does **not** register as a hit there has a grader
that can never fire, and silently lowers every exposure score it appears in.
This is a bug in the corpus, and CI fails on it.

The test suite additionally asserts that every extraction probe plants a canary
and every injection probe carries an untrusted channel — an injection probe
without one is a jailbreak probe filed in the wrong family, and would flatter
every input-side defense measured against it.

## 10. Known limitations

1. **Single-turn.** Multi-turn priming, gradual context poisoning and
   cross-session attacks are not covered. Several real attacks need turns this
   corpus does not have.
2. **Proxy behaviours.** See §2 and §4. Rule-following is measured; harm is not.
3. **The corpus is finite and public.** A target trained on it would score well
   without being safer. Rotate in your own probes for anything load-bearing.
4. **Mock numbers measure the harness.** They exercise graders, statistics and
   reproducibility, and say nothing about any real model.
5. **English and French only.** The multilingual probes are a thin sample, and
   the refusal markers cover two languages.
6. **No cost or latency modelling.** A defense that halves attack success and
   triples latency looks free in this report.

## 11. What a good result looks like

A single exposure score is what gets quoted and hides which family failed, so
read the per-family table first. In practice:

- **Extraction near zero with output filtering on** is expected and not
  impressive — check whether the residual is all obfuscated leaks, because that
  is what the filter cannot see.
- **Injection is the hard family.** A target that scores well there under
  `indirect-rag` and `indirect-tool-output` has done real work.
- **A defense with overlapping intervals has not been shown to work.** Raise
  `--repeat` and re-run before concluding anything.
