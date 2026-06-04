# Nesterov Accelerated Gradient (NAG)

## 1. Problem Setting

We are minimizing a differentiable loss function $J(w)$ with respect to a parameter vector $w \in \mathbb{R}^n$. In this codebase, $J$ is either the **Mean Squared Error** (linear regression) or **Binary Cross-Entropy** (logistic regression), and $w$ are the model weights.

Standard gradient descent follows the steepest descent direction:

$$w_{t+1} = w_t - \eta \nabla J(w_t)$$

where $\eta$ is the learning rate. This converges slowly in ill-conditioned ravines (where the curvature is much higher in one direction than another).

---

## 2. Classical Momentum (Heavy-Ball Method)

Classical momentum (Polyak, 1964) accelerates GD by accumulating a velocity vector that acts as a smoothed gradient:

$$
\begin{aligned}
v_{t+1} &= \beta v_t + \eta \nabla J(w_t) \\
w_{t+1} &= w_t - v_{t+1}
\end{aligned}
$$

with $\beta \in [0,1)$ controlling how much of the previous velocity is retained.

Equivalently, in the form used by `Momentum` in this codebase:

$$
\begin{aligned}
v_{t+1} &= \beta v_t + (1-\beta) \nabla J(w_t) \\
w_{t+1} &= w_t - \eta v_{t+1}
\end{aligned}
$$

(Re-scaling $\eta$ absorbs the $(1-\beta)$ factor.)

**Intuition**: If gradients consistently point in the same direction, velocity builds up and the step size grows. If gradients oscillate, velocity averages them out, damping the oscillation.

---

## 3. Nesterov Accelerated Gradient — The Key Idea

NAG (Nesterov, 1983) modifies classical momentum by computing the gradient **not at the current position** $w_t$, but at a **look-ahead position** that anticipates where the momentum is about to take us.

### 3.1 Standard Formulation

$$
\begin{aligned}
v_{t+1} &= \beta v_t + \eta \nabla J(w_t - \beta v_t) \\
w_{t+1} &= w_t - v_{t+1}
\end{aligned}
$$

### 3.2 Equivalent Two-Step Form (used in this codebase)

The standard form can be rearranged by defining the look-ahead weight $\tilde{w}_t = w_t + \beta v_t$ (note the sign convention varies across sources; this codebase uses the "add" convention):

$$
\begin{aligned}
\tilde{w}_t &= w_t + \beta v_t \quad &\text{(1) look-ahead} \\
g_t &= \nabla J(\tilde{w}_t) \quad &\text{(2) gradient at look-ahead} \\
v_{t+1} &= \beta v_t - \eta g_t \quad &\text{(3) velocity update} \\
w_{t+1} &= w_t + v_{t+1} \quad &\text{(4) weight update}
\end{aligned}
$$

This is **exactly** what the `NAG.Run()` method implements in `Optimizers.cs:82-121`:

| Code line | Mathematical step |
|---|---|
| `wa[j] = w[j] + beta * v[j]` | $\tilde{w} = w_t + \beta v_t$ |
| `grad = model.GradientOne(wa, ...)` | $g_t = \nabla J(\tilde{w}_t)$ |
| `v[j] = beta * v[j] - lr * grad[j]` | $v_{t+1} = \beta v_t - \eta g_t$ |
| `w[j] += v[j]` | $w_{t+1} = w_t + v_{t+1}$ |

### 3.3 Graphical Intuition

```
Classical Momentum:
  w_t ──(velocity βv)──→ w_t - βv_t   (where momentum *would* take us)
  then compute gradient at w_t, apply correction

Nesterov Momentum:
  w_t ──(velocity βv)──→ w_t - βv_t   (peek ahead)
  then compute gradient at peek-ahead point, apply correction from there
```

NAG is often described as having **"peek-ahead"** or **"look-ahead"** because it first takes a jump using the accumulated velocity, *then* evaluates the gradient at that jumped-to point, and finally corrects itself.

---

## 4. Why Does This Help?

### 4.1 Error Correction

Classical momentum can overshoot: if the gradient abruptly changes direction, the velocity carries the optimizer past the minimum. NAG reduces this by evaluating the gradient **after** applying the velocity step. If the look-ahead point overshoots, the gradient will point backwards, and the velocity update will be corrected immediately.

### 4.2 Theoretical Guarantees

For a convex $L$-smooth function, Nesterov achieves the optimal convergence rate for first-order methods:

$$
J(w_t) - J(w^*) = O\left(\frac{1}{t^2}\right)
$$

compared to $O(1/t)$ for standard gradient descent and $O(1/t)$ for classical momentum (no asymptotic improvement over GD for general smooth convex functions in the worst case).

| Method | Convergence rate (convex, smooth) |
|---|---|
| Gradient Descent | $O(1/t)$ |
| Classical Momentum | $O(1/t)$ (same constant) |
| **Nesterov AG** | $\mathbf{O(1/t^2)}$ |

### 4.3 Intuitive Explanation of the $O(1/t^2)$ Rate

NAG can be interpreted as a **gradient descent on a surrogate function** that includes a Bregman divergence term, which effectively gives the optimizer a form of "momentum-adapted preconditioning." The look-ahead step acts as a predictor, and the gradient step acts as a corrector, forming a predictor-corrector scheme that reduces the constant in the convergence bound.

---

## 5. Connection to Ordinary Differential Equations

For small $\eta$, the NAG update approximates the second-order ODE:

$$\ddot{w} + \frac{3}{t} \dot{w} + \nabla J(w) = 0$$

known as the **heavy-ball with friction** ODE. The $\frac{3}{t}\dot{w}$ term provides critical damping that eliminates oscillations while maintaining acceleration. This is in contrast to classical momentum, which approximates:

$$\ddot{w} + \beta \dot{w} + \nabla J(w) = 0$$

where $\beta$ is constant damping — insufficient damping leads to overshooting.

---

## 6. Stochastic NAG (This Codebase's Variant)

This codebase implements **stochastic** NAG: the gradient is computed on a single randomly-shuffled sample at each step, rather than on the full dataset.

$$
\begin{aligned}
\tilde{w}_t &= w_t + \beta v_t \\
g_t &= \nabla J_i(\tilde{w}_t) \quad \text{(gradient on one sample)} \\
v_{t+1} &= \beta v_t - \eta g_t \\
w_{t+1} &= w_t + v_{t+1}
\end{aligned}
$$

The stochastic version does **not** retain the $O(1/t^2)$ rate of full-batch Nesterov (it reverts to $O(1/\sqrt{t})$ for non-convex or $O(1/t)$ for strongly-convex stochastic settings). However, it often converges faster in practice than stochastic classical momentum because the look-ahead still provides better gradient alignment and variance reduction.

---

## 7. Detailed Walkthrough of the Code Path

Given `w`, `v`, learning rate `lr = 0.1`, `beta = 0.9`, and a sample `(x, y)`:

1. **Look-ahead** (lines 101-103):
   $$
   \tilde{w}_j = w_j + 0.9 \cdot v_j \quad \forall j
   $$

2. **Gradient at look-ahead** (line 105):
   $$
   g_j = \frac{\partial J(\tilde{w}, x, y)}{\partial \tilde{w}_j}
   $$
   For logistic regression: $g_j = (\sigma(\tilde{w}^\top x) - y) \cdot x_j$

3. **Velocity update** (line 109):
   $$
   v_j \leftarrow 0.9 \cdot v_j - 0.1 \cdot g_j
   $$

4. **Weight update** (line 110):
   $$
   w_j \leftarrow w_j + v_j
   $$

After all samples in an epoch, loss is logged (line 114-115).

---

## 8. Contrast: Standard Momentum (in this codebase)

For comparison, `Momentum` in line 69-79:

$$
\begin{aligned}
s_{t+1} &= 0.9 \cdot s_t + 0.1 \cdot \nabla J(w_t) \\
w_{t+1} &= w_t - \eta \cdot s_{t+1}
\end{aligned}
$$

Note the $(1-\beta) = 0.1$ scaling on the gradient (exponential moving average formulation). The key difference: momentum evaluates the gradient at $w_t$, NAG evaluates it at $w_t + \beta v_t$.

---

## 9. Hyperparameter $\beta$

- $\beta = 0.9$ is the hardcoded default in this codebase.
- Typical range: $[0.9, 0.99]$.
- Higher $\beta$ gives smoother velocity but more lag.
- In the original Nesterov formulation for convex optimization, $\beta$ is scheduled as $\beta_t = \frac{t-1}{t+2}$, which yields the optimal $O(1/t^2)$ rate. Constant $\beta$ (as used here) is simpler and works well in practice for deep learning.

---

## References

1. Nesterov, Y. (1983). *A method for solving the convex programming problem with convergence rate $O(1/k^2)$*. Doklady ANSSSR.
2. Sutskever, I., et al. (2013). *On the importance of initialization and momentum in deep learning*. ICML.
3. Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods*. USSR Computational Mathematics and Mathematical Physics.
