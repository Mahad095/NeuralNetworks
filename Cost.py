class Cost:
    def __init__(self, CostFunction, CostPrime):
        self.func = CostFunction
        self.prime = CostPrime

    def EvaluateFunction(self, prediction, Actual):
        return self.func(prediction, Actual)

    def EvaluatePrime(self, prediction, Actual):
        return self.prime(prediction, Actual)