using System;

class StochasticGradientDescent
{
    static Random rng = new Random(42);

    // SGD: gradient computed on a single sample
    static double[] GradientOneSample(double[] w, double[] x, double y)
    {
        double error = Predict(w, x) - y;
        return new double[] { error, error * x[0], error * x[1] };
    }

    // Fisher-Yates shuffle — randomizes sample order each epoch
    static void Shuffle(int[] indices)
    {
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }

    static double[] Run(double[][] X, double[] Y, double learningRate, int epochs)
    {
        double[] w = { 0.0, 0.0, 0.0 };
        int N = X.Length;
        int[] indices = new int[N];
        for (int i = 0; i < N; i++) indices[i] = i;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(indices); // new random order each epoch

            foreach (int i in indices)
            {
                double[] grad = GradientOneSample(w, X[i], Y[i]); // one sample at a time
                for (int j = 0; j < w.Length; j++)
                    w[j] -= learningRate * grad[j];                // immediate weight update
            }

            if (epoch % 500 == 0)
                Console.WriteLine($"Epoch {epoch,5}: Loss = {BCE(w, X, Y):F6} | w = [{w[0]:F3}, {w[1]:F3}, {w[2]:F3}]");
        }
        return w;
    }

    static void Main()
    {
        double[][] X = { new[] {0.5, 1.0}, new[] {1.5, 2.0}, new[] {3.0, 1.0}, new[] {4.0, 3.0},
                         new[] {1.0, 0.5}, new[] {2.0, 1.5}, new[] {3.5, 2.5}, new[] {5.0, 2.0} };
        double[]   Y = { 0, 0, 0, 1,
                         0, 0, 1, 1 };

        double[] w = Run(X, Y, learningRate: 0.1, epochs: 5000);
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

// The single conceptual shift — **when** the weights are updated:

// | | Batch GD | Stochastic GD |
// |---|---|---|
// | Gradient computed over | all N samples | 1 sample |
// | Weights updated | once per epoch | N times per epoch |
// | Path to minimum | smooth | noisy but faster |
// | Memory needed | full dataset | one sample |
// ```
// Batch:      w ← w − α · (1/N) Σ ∇L(xᵢ)      // one step per epoch
// SGD:        w ← w − α · ∇L(xᵢ)               // one step per sample