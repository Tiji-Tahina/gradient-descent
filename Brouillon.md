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


```


* What is that state? What does it represent? 
* State of what again? 
* Learning rate? Why is it here ? 
We got the learning rate because we are in a model thing update to do? 

* What is the Math behind the Momentum one? 
I know there is like a goog old  