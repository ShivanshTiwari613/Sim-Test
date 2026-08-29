# Understanding this project — the study docs

These are the "explain it to me from zero" notes for the toy text world model,
in reading order. Each doc stands alone, but they build on each other.

| # | Doc | Question it answers |
|---|---|---|
| 1 | [01-big-picture.md](01-big-picture.md) | What is a world model and what did we build? |
| 2 | [02-env.md](02-env.md) | How does the gridworld itself work? (env.py) |
| 3 | [03-data-generation.md](03-data-generation.md) | Where did the 35,000 rows come from? Why does the agent sometimes freeze? (gen_data.py) |
| 4 | [04-tokens-and-tensors.md](04-tokens-and-tensors.md) | What numbers are actually sent to the model? (input_ids / labels / attention_mask) |
| 5 | [05-training.md](05-training.md) | How does training work? What is the LoRA rank-16 bottleneck? (train.py) |
| 6 | [06-eval.md](06-eval.md) | How do we know it worked? (eval.py, drift curve) |
| 7 | [07-pod-infra.md](07-pod-infra.md) | How the GPU runs happen, and the five failures we hit (podenv/) |
| 8 | [08-number-registry.md](08-number-registry.md) | Every fixed number in the project, decoded |

## Interactive companions (artifacts)

- **World on a String** — run report with a playable copy of the environment:
  https://claude.ai/code/artifact/8bce543a-f281-406c-8f0d-45e8e4668cb6
- **Episode Anatomy** — step-through player for 3 real training episodes:
  https://claude.ai/code/artifact/5f37454b-c92e-4a60-ba28-d6a306e8909d
- **Number Atlas** — every number decoded, with a clickable 81-token inspector:
  https://claude.ai/code/artifact/eee36dc4-9700-4364-a660-70b244f1bae0
- **Inside Qwen** — the machine itself from zero: weights, 896-wide vectors,
  q/k/v/o attention, frozen vs training, the LoRA detour:
  https://claude.ai/code/artifact/43750e9b-59e6-4b55-9aff-442bafc49e15

## The one-paragraph version

A 5×5 gridworld (agent, key, locked door, walls) is described by a single
canonical line of text. We played 1,900 random-ish episodes, wrote down 38,000
`(state, action, next_state)` text triples, and fine-tuned Qwen2.5-0.5B with
LoRA (0.44% of its weights) so that given `STATE: … ACTION: …` it prints the
next state string. It learned the physics almost perfectly: 98% exact-match on
single steps, and when fed its own predictions back for 10 steps it stayed
correct 96% of the time — errors don't compound. The state string *is* the
memory; nothing lives outside the prompt.
