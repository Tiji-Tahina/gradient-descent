using System;

static class Utilities
{
    public static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    public static double Dot(double[] w, double[] x)
    {
        double sum = 0;
        for (int i = 0; i < w.Length; i++)
            sum += w[i] * x[i];
        return sum;
    }

    public static double MSE(double[] yTrue, double[] yPred)
    {
        double sum = 0;
        for (int i = 0; i < yTrue.Length; i++)
        {
            double err = yPred[i] - yTrue[i];
            sum += err * err;
        }
        return sum / yTrue.Length;
    }

    public static double BCE(double[] yTrue, double[] yPred)
    {
        double sum = 0;
        for (int i = 0; i < yTrue.Length; i++)
            sum -= yTrue[i] * Math.Log(yPred[i]) + (1 - yTrue[i]) * Math.Log(1 - yPred[i]);
        return sum / yTrue.Length;
    }

    public static void Shuffle<T>(T[] array, Random rng)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
