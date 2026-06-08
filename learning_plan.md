# ML from Scratch: 10-Day Learning Plan

## Structure

Each day follows the **Study → Rewrite → Extend** cycle:
1. **Study** — Read the file, derive the math on paper
2. **Rewrite** — Delete the file and rewrite it from memory without peeking
3. **Extend** — Add a small new capability on top of what you just wrote

---

## Phase 1: Foundation (Days 1–3)

### Day 1 — `Utilities.cs`

| | |
|---|---|
| **Concepts** | Sigmoid, dot product, MSE, BCE, Fisher-Yates shuffle |
| **Math to derive** | • Sigmoid derivative: $\sigma'(z) = \sigma(z)(1-\sigma(z))$<br>• MSE gradient: $\frac{\partial}{\partial w_j} (w^T x - y)^2 = 2(w^T x - y) x_j$<br>• BCE gradient: $\frac{\partial}{\partial w_j} [y\log(\hat{y}) + (1-y)\log(1-\hat{y})] = (\sigma(w^T x) - y) x_j$ |
| **Extension** | Add `Relu` and `Tanh` activation functions |
| **Files involved** | `Utilities.cs` |

**Rewrite target** (~45 lines):

```csharp
using System;

static class Utilities
{
    public static double Sigmoid(double z) { ... }
    public static double Dot(double[] w, double[] x) { ... }
    public static double MSE(double[] yTrue, double[] yPred) { ... }
    public static double BCE(double[] yTrue, double[] yPred) { ... }
    public static void Shuffle<T>(T[] array, Random rng) { ... }
}
```

---

### Day 2 — `Models.cs`

| | |
|---|---|
| **Concepts** | `IModel` interface, `LinearModel` (MSE), `LogisticModel` (BCE), full-gradient vs. per-sample gradient |
| **Math to derive** | • Full-batch MSE gradient: $\nabla_w \text{MSE} = \frac{2}{n} X^T (Xw - y)$<br>• Full-batch BCE gradient: $\nabla_w \text{BCE} = \frac{1}{n} X^T (\sigma(Xw) - y)$<br>• Why `GradientOne` drops the $\frac{2}{n}$ or $\frac{1}{n}$ factor (the optimizer/loop handles averaging) |
| **Extension** | Add `RidgeModel` with L2 regularization: Loss = MSE + $\lambda\|w\|^2$. Gradient gains a $2\lambda w_j$ term for all $j$ except the bias. |
| **Files involved** | `Models.cs` |

**Key interface**:

```csharp
interface IModel
{
    double Predict(double[] w, double[] x);
    double Loss(double[] w, double[][] X, double[] y);
    double[] Gradient(double[] w, double[][] X, double[] y);
    double[] GradientOne(double[] w, double[] x, double y);
}
```

---

### Day 3 — `data.cs` + `.csproj`

| | |
|---|---|
| **Concepts** | Bias-as-feature trick, hardcoded datasets, `.csproj` structure |
| **Math to derive** | • Why $w_0 + w_1 x_1 = [1, x_1] \cdot [w_0, w_1]$ — the bias column lets the optimizer learn the intercept naturally |
| **Extension** | Create a non-linear dataset (e.g., $y = x^2 + \text{noise}$) and see what happens when you fit a linear model to it |
| **Files involved** | `data.cs`, `GradientDescent.csproj` |
| **Known fix** | Rename `data.cs` → `Data.cs` for PascalCase consistency; add missing `using System;` |

**Data layout** (first column is always 1 for bias):

```
Linear:  X = [[1,1,2], [1,2,3], [1,3,4], [1,4,5], [1,5,6]]
         y = [3, 5, 7, 9, 11]     (≈ 2*x₁ + 3)

Logistic: X = [[1,1,2], [1,2,3], [1,3,4], [1,4,5], [1,5,6]]
          y = [0, 0, 0, 1, 1]     (binary classification)
```

---

## Phase 2: Optimizers (Days 4–7)

### Day 4 — `BaseOptimizer` + `GD`

| | |
|---|---|
| **Concepts** | Template Method pattern, full-batch gradient descent, `virtual`/`override`, `abstract` |
| **Math to derive** | • GD update: $w_{t+1} = w_t - \eta \nabla f(w_t)$<br>• Convergence $O(1/t)$ for convex $L$-smooth functions<br>• Why step size must satisfy $\eta < 2/L$ for convergence |
| **Extension** | Add learning rate decay: `lr = lr0 / (1 + decay * epoch)` |
| **Files involved** | `Optimizers.cs` |

**Template Method skeleton**:

```csharp
abstract class BaseOptimizer
{
    public virtual double[] Run(IModel model, double[][] X, double[] y,
                                double lr, int epochs)
    {
        // Initialize w, state; loop epochs: shuffle, loop samples,
        // call GradientOne, call Step, log loss
    }

    protected abstract void Step(double[] w, double[] grad,
                                 ref double[] state, double lr);
}
```

GD overrides `Run()` to use full-batch `Gradient()` instead of per-sample `GradientOne()`. Its `Step()` is just:

```csharp
w[j] -= lr * grad[j];
```

---

### Day 5 — `SGD`

| | |
|---|---|
| **Concepts** | Per-sample gradient, stochasticity, unbiasedness, trade-offs |
| **Math to derive** | • $\mathbb{E}[\nabla f_i(w)] = \nabla f(w)$ — each sample's gradient is an unbiased estimate of the true gradient<br>• Variance: $\text{Var}(\nabla f_i(w)) = \frac{1}{n}\sum\|\nabla f_i(w) - \nabla f(w)\|^2$ — noisy but cheap |
| **Extension** | Implement **Mini-batch GD**: sum gradients over a batch of size $b$ (parameter), then update. Your batch size becomes a hyperparameter between 1 (SGD) and $n$ (GD). |
| **Files involved** | `Optimizers.cs` |

**SGD is the simplest** — it inherits `Run()` from `BaseOptimizer` unchanged. Its `Step()` is identical to GD's:

```csharp
w[j] -= lr * grad[j];
```

The only difference is that `Run()` calls `GradientOne` (per sample) instead of `Gradient` (full batch).

---

### Day 6 — `Momentum`

| | |
|---|---|
| **Concepts** | Velocity buffer, exponential moving average, smoothing oscillations, escaping local minima |
| **Math to derive** | • Momentum update:<br>  $v_{t+1} = \beta v_t + \nabla f(w_t)$<br>  $w_{t+1} = w_t - \eta v_{t+1}$<br>• $v_t$ is an EMA of past gradients: $v_t = \sum_{k=0}^{t} \beta^{t-k} \nabla f(w_k)$<br>• $\beta = 0.9$ means ~10 steps of memory |
| **Extension** | Implement **RMSprop**: `state[j] = decay * state[j] + (1-decay) * grad[j]²`, then `w[j] -= lr * grad[j] / (sqrt(state[j]) + eps)` |
| **Files involved** | `Optimizers.cs` |

**Momentum's `Step()`** uses `state` as the velocity buffer:

```csharp
double beta = 0.9;
state[j] = beta * state[j] + (1 - beta) * grad[j];
w[j] -= lr * state[j];
```

---

### Day 7 — `NAG` (Nesterov Accelerated Gradient)

| | |
|---|---|
| **Concepts** | Look-ahead gradient, ODE connection, optimal convergence $O(1/t^2)$ |
| **Math to derive** | • NAG update:<br>  $w_{\text{lookahead}} = w_t + \beta v_t$<br>  $v_{t+1} = \beta v_t + \nabla f(w_{\text{lookahead}})$<br>  $w_{t+1} = w_t - \eta v_{t+1}$<br>• Compare to Momentum: gradient is evaluated *after* the look-ahead step, not at current $w_t$<br>• Convergence: $f(w_t) - f(w^*) \leq O(1/t^2)$ vs. Momentum's $O(1/t)$ for convex smooth functions |
| **Extension** | Implement **Adam**: combines Momentum (`state` for velocity) + RMSprop (second `state` for squared gradients) + bias correction |
| **Files involved** | `Optimizers.cs` |
| **Known fix** | `NAG.Step()` is currently empty — NAG overrides `Run()` entirely instead. Refactor so NAG uses the base `Run()` with a proper `Step()`. |

**NAG currently does this** (override `Run` directly):

```csharp
// Look-ahead
wa[j] = w[j] + beta * v[j];
// Gradient at look-ahead point
grad = model.GradientOne(wa, X[i], Y[i]);
// Update velocity and weights
v[j] = beta * v[j] - lr * grad[j];
w[j] += v[j];
```

**Refactoring challenge**: Move the look-ahead logic into `Step()` so the base `Run()` can be reused. You'll need to pre-compute $w + \beta v$ before calling `GradientOne`, then call `Step()` with the result.

---

## Phase 3: Integration (Days 8–10)

### Day 8 — `Program.cs`

| | |
|---|---|
| **Concepts** | Orchestration, hyperparameter tuning, debugging convergence |
| **Activities** | 1. Study the entry point — 5 demos using 2 models × 4 optimizers<br>2. Rewrite from scratch<br>3. Add diagnostic output: weight trajectory every N epochs, loss curve visualization (console histogram) |
| **Files involved** | `Program.cs` |

**Current demo lineup**:

| # | Model | Optimizer | LR | Epochs |
|---|-------|-----------|----|--------|
| 1 | Linear | GD | 0.01 | 1000 |
| 2 | Logistic | GD | 0.1 | 2000 |
| 3 | Logistic | SGD | 0.1 | 2000 |
| 4 | Logistic | Momentum | 0.1 | 2000 |
| 5 | Logistic | NAG | 0.1 | 2000 |

---

### Day 9 — Full Project Rewrite (Blank Slate)

**All files, zero references, from memory.**

```
dotnet new console -n GradientDescent
cd GradientDescent
```

Write every file in this order:
1. `Utilities.cs`
2. `Models.cs`
3. `Data.cs`
4. `Optimizers.cs`
5. `Program.cs`

Build and run after each file. At the end, compare output with the original (`git diff` + output diff).

---

### Day 10 — Extension into New Territory

Pick **one** (or more):

| Option | Difficulty | Description |
|--------|-----------|-------------|
| **A. Multiclass classification** | Hard | Implement softmax + cross-entropy loss; use Iris dataset (3 classes) |
| **B. Neural network (1 hidden layer)** | Hard | Build a `NeuralNetwork` class implementing `IModel`; one hidden layer with ReLU, output layer with sigmoid/softmax |
| **C. Console visualizations** | Medium | Plot loss curves, weight trajectories, or decision boundaries using ASCII |
| **D. Optimizer benchmark** | Medium | Run all 4 optimizers on the same problem, log loss per epoch, plot convergence comparison |

---

## How to Execute Each Day

```bash
# 1. Read the file thoroughly
cat Models.cs              # (or open in editor)

# 2. Derive the math on paper
#    Write out gradients, loss functions, update rules by hand

# 3. Close the file — no peeking

# 4. Rename the original to force yourself to recreate it
mv Models.cs Models.cs.bak

# 5. Write from memory
#    Compile after every function
dotnet build

# 6. Diff against your original
git diff --no-index Models.cs.bak Models.cs

# 7. Do the extension
#    Add the new feature on your own

# 8. Commit
git add -A && git commit -m "Day N: ..."
```

### Build & Verify

```bash
dotnet build   # zero errors
dotnet run     # loss must decrease, weights must converge
```

---

## Existing Issues to Fix as You Go

| Issue | File | Day | Fix |
|-------|------|-----|-----|
| `Utilities.MSE()` and `BCE()` unused | `Utilities.cs` | 1 | Delete them or refactor models to call them |
| `NAG.Step()` is empty (inheritance misuse) | `Optimizers.cs` | 7 | Refactor NAG to use proper `Step()` override |
| `data.cs` (lowercase d) | — | 3 | Rename to `Data.cs` |
| Missing `using System;` | `data.cs` | 3 | Add it |

---

## Quick Reference: Key Math Formulas

| Concept | Formula |
|---------|---------|
| Linear prediction | $\hat{y} = w^T x$ |
| Sigmoid | $\sigma(z) = 1 / (1 + e^{-z})$ |
| MSE | $\frac{1}{n} \sum (\hat{y}_i - y_i)^2$ |
| BCE | $-\frac{1}{n} \sum [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$ |
| GD update | $w \leftarrow w - \eta \nabla f(w)$ |
| SGD update | $w \leftarrow w - \eta \nabla f_i(w)$ (one sample) |
| Momentum | $v \leftarrow \beta v + (1-\beta)\nabla f(w)$; $w \leftarrow w - \eta v$ |
| NAG | $w_a = w + \beta v$; $v \leftarrow \beta v + \nabla f(w_a)$; $w \leftarrow w - \eta v$ |
| Fisher-Yates | For $i = n-1 \dots 1$: swap `array[i]` with `array[rng.Next(i+1)]` |
