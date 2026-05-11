using System;

class GradientDescent
{
    // Sigmoid activation: σ(z) = 1 / (1 + e⁻ᶻ)
    static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));

    // Sigmoid derivative: σ'(z) = σ(z) · (1 − σ(z))
    static double SigmoidDerivative(double z) { double s = Sigmoid(z); return s * (1 - s); }

    // Model: ŷ = σ(w0 + w1*x1 + w2*x2)
    static double LinearCombination(double[] w, double[] x) => w[0] + w[1] * x[0] + w[2] * x[1];
    static double Predict(double[] w, double[] x) => Sigmoid(LinearCombination(w, x));

    // Loss: Binary Cross-Entropy = -(1/N) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
    static double BCE(double[] w, double[][] X, double[] Y)
    {
        double loss = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double yHat = Predict(w, X[i]);
            loss -= Y[i] * Math.Log(yHat) + (1 - Y[i]) * Math.Log(1 - yHat);
        }
        return loss / X.Length;
    }

    // Gradient of BCE w.r.t. each weight (via chain rule):
    // ∂L/∂wⱼ = (1/N) * Σ (ŷ - y) * xⱼ     (bias: xⱼ = 1)
    static double[] Gradient(double[] w, double[][] X, double[] Y)
    {
        double[] grad = new double[3];
        for (int i = 0; i < X.Length; i++)
        {
            double error = Predict(w, X[i]) - Y[i]; // (ŷ - y): BCE + sigmoid simplify cleanly
            grad[0] += error;            // ∂L/∂w0 (bias)
            grad[1] += error * X[i][0]; // ∂L/∂w1
            grad[2] += error * X[i][1]; // ∂L/∂w2
        }
        for (int j = 0; j < grad.Length; j++)
            grad[j] /= X.Length;
        return grad;
    }

    static double[] Run(double[][] X, double[] Y, double learningRate, int iterations)
    {
        double[] w = { 0.0, 0.0, 0.0 };

        for (int i = 0; i < iterations; i++)
        {
            double[] grad = Gradient(w, X, Y);
            for (int j = 0; j < w.Length; j++)
                w[j] -= learningRate * grad[j]; // w ← w − α · ∇L

            if (i % 500 == 0)
                Console.WriteLine($"Step {i,5}: Loss = {BCE(w, X, Y):F6} | w = [{w[0]:F3}, {w[1]:F3}, {w[2]:F3}]");
        }
        return w;
    }

    static void Main()
    {
        // Binary classification dataset (linearly separable)
        double[][] X = { new[] {0.5, 1.0}, new[] {1.5, 2.0}, new[] {3.0, 1.0}, new[] {4.0, 3.0},
                         new[] {1.0, 0.5}, new[] {2.0, 1.5}, new[] {3.5, 2.5}, new[] {5.0, 2.0} };
        double[]   Y = { 0, 0, 0, 1,
                         0, 0, 1, 1 };

        double[] w = Run(X, Y, learningRate: 0.1, iterations: 5000);
        Console.WriteLine($"\nLearned weights: w0={w[0]:F3}, w1={w[1]:F3}, w2={w[2]:F3}");

        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < X.Length; i++)
        {
            double yHat = Predict(w, X[i]);
            Console.WriteLine($"  x=[{X[i][0]}, {X[i][1]}] → ŷ={yHat:F3}  (label={Y[i]})");
        }
    }
}
// ```

// ---

// ### What changed

// Two things were introduced:

// **1. Sigmoid replaces the identity output**
// ```
// Before:  ŷ =      w₀ + w₁x₁ + w₂x₂       → ŷ ∈ (-∞, +∞)
// After:   ŷ = σ(  w₀ + w₁x₁ + w₂x₂  )     → ŷ ∈ (0, 1)
// ```

// **2. BCE replaces MSE** — the natural loss for probabilistic outputs:
// ```
// L = − [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
// ```

// **The elegant cancellation** — chain rule through BCE + sigmoid collapses neatly:
// ```
// ∂L/∂wⱼ = (ŷ − y) · xⱼ