namespace utilities
{
    public class Functions
    {
        public double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public double SigmoidDerivative(double z)
        {
            double s = Sigmoid(z);
            return s * (1 - s);
        }

        public double LinearCombination(double[] w, double[] x) // just for one point okay 
        {
            return w[0] + w[1] * x[0] + w[2] * x[1];
        }

        public double Predict(double[] w, double[] x)
        {
            return Sigmoid(LinearCombination(w, x));
        }

        public double BCE(double[] w, double[][] X, double[] Y)
        {
            double loss = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double yPred = Predict(w, X[i]);
                loss -= Y[i] * Math.Log(yPred) + (1 - Y[i]) * Math.Log(1 - yPred);
            }
            return loss / X.Length;
        }

        public double MSE(double[] w, double[][] x, double[] y)
        {
            double loss = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double yPred = Predict(w, x[i]);
                loss += Math.Pow(yPred - y[i]);
            }
        }
    }
}