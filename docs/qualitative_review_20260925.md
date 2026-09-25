# Qualitative review of representative main-run debates — 25 September 2026

## Purpose and limits

The numerical evaluation establishes what changed between rounds. This review
examines stored reasoning to describe how some changes occurred. It is a
post-result, descriptive analysis, not a new experiment or a causal test. No
model was called, no answer was regenerated, and no stored row was changed.

The selection was made transparently:

- all four complete-case questions where Round 1 already had a majority and
  correctness changed: the two improvements and both regressions;
- two complete-case improvements that began without a Round 1 majority;
- one complete-case example of convergence on a wrong answer; and
- one complete-case example where disagreement persisted.

This set is deliberately balanced. It includes successful correction,
successful resolution of disagreement, harmful persuasion, convergence without
correctness and no effect. The benchmark key remains the scoring authority even
where the stored responses expose plausible ambiguity in a question.

## Summary

| Question | Key | Round 1 | Round 2 | What the stored reasoning shows |
|---|---:|---|---|---|
| `test:5503` | A | Wrong consensus C | Unanimous A | A correct minority supplied the decisive definition |
| `test:7670` | H | Wrong consensus I | Correct consensus H | Peers corrected a concrete accounting error |
| `test:11081` | E | No consensus | Unanimous E | A truth table exposed ignored premises |
| `test:5605` | J | No consensus | Unanimous J | Two agreeing factual claims attracted the other agents |
| `test:5104` | I | Correct consensus I | No consensus | One correct voter adopted a confident fabricated film memory |
| `test:7075` | F | Correct consensus F | Wrong consensus I | A plausible progressive-tax interpretation displaced the keyed interpretation |
| `test:1203` | I | No consensus | Unanimous D | Agents converged on a timing-based legal reading that conflicts with the key |
| `test:2697` | E | No consensus | No consensus | Each side repeated its initial general knowledge; no decisive evidence emerged |

## 1. Correcting an existing wrong majority

### `mmlu_pro_v1:test:5503` — securitization theory

- **Correct answer:** A
- **Round 1:** C won 3–2. DeepSeek, Llama and Mistral chose C; Gemma and Qwen
  chose A.
- **Round 2:** all five chose A.

The disagreement concerned the exact threshold for successful securitization.
The A responses argued that the process fails when the audience does not grant
the actor the right to use extraordinary measures. The C responses initially
treated implementation of those measures as necessary.

In Round 2 the three C voters explicitly adopted the distinction made by the A
responses: audience acceptance establishes securitization; later failure to
implement a measure is a policy or implementation failure. Here peer reasoning
corrected a specific conceptual error and overturned a three-agent wrong
majority.

### `mmlu_pro_v1:test:7670` — bank reserves and money supply

- **Correct answer:** H, a `$30,000` reduction
- **Round 1:** I won 3–2. DeepSeek, Llama and Mistral calculated `$40,000`;
  Gemma and Qwen calculated `$30,000`.
- **Round 2:** H won 4–1. DeepSeek and Mistral changed to H; Llama kept I.

Both camps recognised the four-times deposit contraction. The decisive point
was that the withdrawn `$10,000` remained in the money supply as currency.
Therefore deposits contracted by `$40,000`, currency increased by `$10,000`,
and the net money-supply reduction was `$30,000`.

DeepSeek and Mistral explicitly identified the missing currency offset in their
initial calculations. This is a strong example of debate helping when a peer
provides a checkable correction rather than merely a different letter.

## 2. Resolving initial disagreement correctly

### `mmlu_pro_v1:test:11081` — propositional logic

- **Correct answer:** E, valid
- **Round 1:** no consensus: E received two votes; A, F and I received one each.
- **Round 2:** unanimous E.

DeepSeek and Gemma correctly checked only truth-table rows where both premises
were true. The three wrong responses each proposed a counterexample but ignored
one of the premises. In Round 2, Llama and Qwen explicitly acknowledged that
their proposed rows did not satisfy every premise, and Mistral performed the
full check again.

The debate resolved disagreement because the correct reasoning was directly
verifiable. The peers did not need to trust a model's memory; they could repeat
the truth-table test.

### `mmlu_pro_v1:test:5605` — 2013 corruption survey figure

- **Correct answer:** J, `76%`
- **Round 1:** no consensus: J received two votes; C, F and H received one each.
- **Round 2:** unanimous J.

DeepSeek and Qwen independently selected 76% and referred to Gallup-style
polling. The other three agents initially proposed 66%, 56% and 90%. In Round 2
they moved to J because two peers supplied the same figure and described it as
the commonly cited statistic.

This is a weaker explanatory case than the truth-table example. The stored text
contains claims about external polling but cannot verify those sources. The
answer became correct, but the mechanism may combine knowledge sharing with
social reinforcement. The result supports successful convergence, not proof
that the agents independently validated the statistic.

## 3. Losing a correct Round 1 result

### `mmlu_pro_v1:test:5104` — *The Hudsucker Proxy*

- **Correct answer:** I, Hula Hoop
- **Round 1:** I won 3–1–1. Gemma chose E (Slinky) and Qwen chose H (Yo-Yo).
- **Round 2:** no consensus: I had two votes, H had two and E had one.

DeepSeek, Llama and Mistral initially knew that the film centres on the Hula
Hoop. Qwen confidently supplied false details about a toy called the “Wimp” and
misidentified actors and characters; Gemma supplied a different fabricated
telephone/Slinky plot. DeepSeek explicitly rejected both stories in Round 2,
but Mistral changed from the correct Hula Hoop answer to Qwen's Yo-Yo answer.

This is the clearest harmful-persuasion example. A fluent and specific peer
account sounded authoritative despite being false. Debate did not make the two
remaining correct agents wrong, but one switch was enough to destroy the group
majority.

### `mmlu_pro_v1:test:7075` — tax brackets

- **Correct answer under the benchmark key:** F
- **Round 1:** F won 3–1–1. DeepSeek chose H and Qwen chose I.
- **Round 2:** I won 3–2. Llama and Mistral changed from F to I; DeepSeek
  changed from H to F.

The F reasoning applies the stated 25% rate to the full `$210`: take-home pay
falls from `$160` to `$157.50`, and the additional `$12.50` tax on a `$10`
raise produces the option's effective marginal rate of 125%. Qwen instead used
the normal real-world interpretation of progressive brackets, taxing only the
extra `$10` at 25% and obtaining a `$7.50` increase.

Llama and Mistral found the progressive interpretation more economically
plausible and changed to I. The benchmark nevertheless keys F. This regression
therefore exposes question-framing tension: peer reasoning can favour a
reasonable real-world interpretation that does not match the benchmark's
intended simplified calculation. It should be discussed as a limitation, not
presented simply as irrational model behaviour.

## 4. Consensus is not the same as correctness

### `mmlu_pro_v1:test:1203` — lease termination

- **Correct answer under the benchmark key:** I
- **Round 1:** no consensus: D and I had two votes each; F had one.
- **Round 2:** unanimous D, which is wrong under the key.

The D responses focused on the hypothetical timing: eviction immediately after
the fixed term and before acceptance of the next rent payment. They argued that
no periodic tenancy yet existed, so a verbal extension was the tenant's
strongest possible argument. The I responses initially focused on the earlier
statement that rent had continued and been accepted, implying a periodic
tenancy requiring notice.

All five eventually adopted the timing-based D interpretation. The stored
reasoning is coherent, but it conflicts with the benchmark key. This case shows
why the project scores against the fixed key while also inspecting the text:
unanimity can reflect shared interpretation of an ambiguous item rather than
truth.

## 5. Debate sometimes changes nothing

### `mmlu_pro_v1:test:2697` — work-related accidents

- **Correct answer:** E, more recent life stress
- **Round 1:** no consensus: E and C had two votes each; A had one.
- **Round 2:** exactly the same split.

The E agents cited the general stress–accident relationship. The C agents
argued that accident-prone workers receive fewer promotions. Qwen asserted the
opposite promotion relationship and kept A. In Round 2 every agent defended its
initial knowledge claim; none supplied a calculation, definition or concrete
piece of evidence capable of resolving the conflict.

This demonstrates that exposing agents to more text does not guarantee either
agreement or correction. When the disagreement depends on recalled empirical
facts and no response provides verifiable evidence, Round 2 may simply preserve
the original split.

## Cross-case findings

1. **Checkable reasoning produced the clearest improvements.** The truth table,
   reserve accounting and formal definition gave agents something they could
   inspect and correct.
2. **Confident detail can spread an error.** The film regression began when one
   correct voter adopted a specific but fabricated peer account.
3. **Plausible alternative interpretations matter.** The tax and lease cases
   show tension between natural real-world readings and the benchmark key.
4. **Debate is strongest at changing agreement.** It can create correct
   consensus, wrong consensus, destroy an existing majority or leave the split
   untouched.
5. **Vote counts alone do not explain the mechanism.** The stored reasoning is
   necessary to distinguish mathematical correction, factual imitation,
   ambiguity and unsupported confidence.

These cases support the quantitative conclusion without replacing it: the
overall frozen system improved by 3.33 percentage points mainly through
questions without an initial majority, but communication was not uniformly
beneficial and increased agreement more reliably than correctness.
