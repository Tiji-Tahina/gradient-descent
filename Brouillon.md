```
class Momentum : BaseOptimizer{
    protectedd override voidd Step(double[] w, double[] grad, ref double[] state, double lr)

    protected
}
```

```class Momenntum : BaseOptimizer{
    protected override void Step(double[] w, double[] grad, ref double[] state, double lr)

classs Momentum : BaseOptimizer {
    protected override voidd Step(double[] w, double[] grad, double[] state, double lr)
}
}

{
    double beta = 0.9; 
    for (int j = 0; j < x.Length; j++){
        state[j] = beta * state[j] + (1 - beta) * grad[j];
        w[j] -= lr * state[j];
    }
}

class NAG : BaseOptimizer{
    public override double[] Run(IModel model, double[][] X, double[] Y, double lr, int epochs){
        int n = X[0].Length ;
        double[] w = new double[n];
        double[] v = new double[n]; 
        int[] indices = new int [X.Length];
        double beta = 0.9;
    }
}


class NAG  : BasOptimizer{
    public override double[] Run ( IModel model, double[]=[] X, double lr, double[] Y, int epochs){
        int n = X[0].Length;
        double[] w = new double[n];
        double[] v = new double[n]; 
        var rng = new Random(0);
        double beta = 0.9;'

        for (int epoch = 0; epoch < epochs; ++epoch){
            for (int i = 0 ; i < indices.Lenght; ++i) indices[i] = i;
            Utilities.Shuffle(indices, rng);

            for (int si = 0 ; si < X.Lenght; si++){]
                int i = indices [si];
                double [] wa = new double [nn];

                # What does this do in itself ? \]\


                epoch is doing his thing

            }
        }
    }
}

```


* What is that state? What does it represent? 
* State of what again? 
* Learning rate? Why is it here ? 
We got the learning rate because we are in a model thing update to do? 

* What is the Math behind the Momentum one? 
I know there is like a goog old  


* epoch is doing the passes we need 
* i going through the data I bet 
* Shuflfe the indices and get them to be usable 

* beta is just a constant that is used to do things in the formula 

* After updating the indices, what do we actually do to deserve anything back? 
```
double[] wa = new double[n]; 
for (int j = 0; j < n; j++)
    wa[j] = w[j] + beta * v[j]; 

double[] grad = modle.GradientOne(wa, X[i-], Y[i])

double[] grad = model.GradientOne(wa, X[i], Y[i])

model.

public static MSE(double[][] X, double[] Yhat, double[] Y) { ... }

class Momentum : BaseOptimizer{
    protected override void Step (double[] w, double[] grad, double lr){
        double beta = 0.9;
        for (int j = 0; j < w.Length; ++j){
            state[j] = beta * state[j] + (1 - beta) * grad[j];
            w[j] -= lr * state[j];
        }
    }
}


class NAG : BaseOptimizer 
{
    public override double[] Run
}
```

* Why is this one on the Base Optimizer called Run and not something else like step in the Momentum one?

* They all have the public virtual double[] Run method that is making 

* double[] wa = new double[n];

```
abstract class BaseOptimizer
{
    public virtual double[] Run(IModel model, double[][] X,  double[] Ym double lr, double[] state, ){
        for (int i = 0; i < epochs; ++i){
            for (x : X){
                
            }
        }
    }
}
```
```
using System;

static class Utilities{
    static void Shuffle<T>(T[] array, Random rng){
        for (i = array.Length() - 1; i > 0; i--){
            int j = rng.Next(i + 1);
            (array[i], array[j]) = ( array[j], array[i]);
        }
    }
}

```