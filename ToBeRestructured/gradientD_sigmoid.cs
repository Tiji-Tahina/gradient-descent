class GradientDescent
{
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