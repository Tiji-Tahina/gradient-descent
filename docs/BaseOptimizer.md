# BaseOptimizer

## What it is

`BaseOptimizer` is an **abstract class** in `Optimizers.cs:3` that implements the [Template Method](https://en.wikipedia.org/wiki/Template_method_pattern) design pattern. It defines the skeleton of a gradient-descent training loop and lets subclasses plug in only the weight-update rule.

## Role in the architecture

| Component | Responsibility |
|---|---|
| `IModel` | Defines the model: `Predict`, `Loss`, `Gradient`, `GradientOne` |
| **`BaseOptimizer`** | Defines the training loop: init weights, shuffle data, iterate epochs + samples, call gradient + step, log loss |
| `GD`, `SGD`, `Momentum`, `NAG` | Only implement the **weight-update rule** via `Step()` |

## The two key methods

### `Run(...)` — the training loop (virtual)

```csharp
public virtual double[] Run(IModel model, double[][] X, double[] Y, double lr, int epochs)
```

1. Initializes weight vector `w` and a `state` vector (both length = number of features).
2. For each epoch: shuffles sample indices, loops over all samples, computes gradient for one sample via `model.GradientOne()`, and calls `Step()`.
3. Logs loss every 100 epochs and at the end.
4. Returns the learned weights.

**Note:** `BaseOptimizer.Run()` uses a *per-sample* loop (SGD-style). `GD` overrides `Run()` to use a *full-batch* gradient (`model.Gradient()` instead of `GradientOne`). This is a design smell — the batch vs. stochastic distinction leaks into the override.

### `Step(...)` — the weight update (abstract)

```csharp
protected abstract void Step(double[] w, double[] grad, ref double[] state, double lr);
```

Each subclass writes only this method:

| Subclass | Update rule |
|---|---|
| `GD` | `w -= lr * grad` (full-batch, overrides `Run` too) |
| `SGD` | `w -= lr * grad` (per-sample) |
| `Momentum` | `v = β·v + (1-β)·grad`; `w -= lr * v` |
| `NAG` | Look-ahead `w + β·v`, gradient at look-ahead, then momentum update (overrides `Run` entirely) |

### The `state` array

`state` is an extra parameter of length `n` passed into every `Step` call. It lets optimizers that maintain internal state (e.g. Momentum's velocity vector, Adam's moment estimates) store that state without adding new fields. Optimizers that don't need it (GD, SGD) simply ignore it.

## How duplication is eliminated

Before this abstraction, every optimizer file had its own copy of the entire training loop (shuffle, epoch loop, logging, etc.). `BaseOptimizer` centralises all of that into one place — each subclass is typically 5–15 lines instead of 50+.

## Design notes / criticisms

- **Batch vs. stochastic leak:** The base `Run` assumes per-sample gradient calls. `GD` and `NAG` override `Run` entirely, which duplicates loop logic. A cleaner design might extract "how to compute gradient" (full-batch vs per-sample) as another abstract method.
- **`NAG.Step` is empty** — NAG overrides `Run` completely and never calls `Step`. This violates Liskov substitution.
- **State array type:** Using `double[]` for state is flexible but type-unsafe (no compile-time check that a given optimizer uses state correctly).
