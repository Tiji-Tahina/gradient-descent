using System;

abstract class BaseOptimizer
{
    public virtual double[] Run(IModel model, double[][] X, double[] Y, double lr, int epochs)
    {
        int n = X[0].Length;
        double[] w = new double[n];
        double[] state = new double[n]; // for Momentum and Adam
        var rng = new Random(0);
        int[] indices = new int[X.Length];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < indices.Length; i++) indices[i] = i;
            Utilities.Shuffle(indices, rng);

            for (int si = 0; si < X.Length; si++)
            {
                int i = indices[si];
                double[] grad = model.GradientOne(w, X[i], Y[i]);
                Step(w, grad, ref state, lr);
            }

            if (epoch % 100 == 0 || epoch == epochs - 1)
                Console.WriteLine($"Epoch {epoch}, Loss: {model.Loss(w, X, Y):F6}");
        }
        return w;
    }

    protected abstract void Step(double[] w, double[] grad, ref double[] state, double lr);
}

class GD : BaseOptimizer
{
    public override double[] Run(IModel model, double[][] X, double[] Y, double lr, int epochs)
    {
        int n = X[0].Length;
        double[] w = new double[n];
        double[] state = new double[n];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double[] grad = model.Gradient(w, X, Y);
            Step(w, grad, ref state, lr);

            if (epoch % 100 == 0 || epoch == epochs - 1)
                Console.WriteLine($"Epoch {epoch}, Loss: {model.Loss(w, X, Y):F6}");
        }
        return w;
    }

    protected override void Step(double[] w, double[] grad, ref double[] state, double lr)
    {
        for (int j = 0; j < w.Length; j++)
            w[j] -= lr * grad[j];
    }
}

class SGD : BaseOptimizer
{
    protected override void Step(double[] w, double[] grad, ref double[] state, double lr)
    {
        for (int j = 0; j < w.Length; j++)
            w[j] -= lr * grad[j];
    }
}

class Momentum : BaseOptimizer
{
    protected override void Step(double[] w, double[] grad, ref double[] state, double lr)
    {
        double beta = 0.9;
        for (int j = 0; j < w.Length; j++)
        {
            state[j] = beta * state[j] + (1 - beta) * grad[j];
            w[j] -= lr * state[j];
        }
    }
}

class NAG : BaseOptimizer
{
    public override double[] Run(IModel model, double[][] X, double[] Y, double lr, int epochs)
    {
        int n = X[0].Length;
        double[] w = new double[n];
        double[] v = new double[n];
        var rng = new Random(0);
        int[] indices = new int[X.Length];
        double beta = 0.9;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < indices.Length; i++) indices[i] = i;
            Utilities.Shuffle(indices, rng);

            for (int si = 0; si < X.Length; si++)
            {
                int i = indices[si];
                double[] wa = new double[n];
                for (int j = 0; j < n; j++)
                    wa[j] = w[j] + beta * v[j];

                double[] grad = model.GradientOne(wa, X[i], Y[i]);

                for (int j = 0; j < n; j++)
                {
                    v[j] = beta * v[j] - lr * grad[j];
                    w[j] += v[j];
                }
            }

            if (epoch % 100 == 0 || epoch == epochs - 1)
                Console.WriteLine($"Epoch {epoch}, Loss: {model.Loss(w, X, Y):F6}");
        }
        return w;
    }

    protected override void Step(double[] w, double[] grad, ref double[] state, double lr) { }
}
