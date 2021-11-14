class CustomCriterion1:

    def __init__(self):
        self.beta = 1

    def calculate_loss(self, inputs, out, mu, sigma):
        return 0.0

    def __str__(self):
        return f'CustomCriterion1:'
