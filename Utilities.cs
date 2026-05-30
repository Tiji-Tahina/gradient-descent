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
}
