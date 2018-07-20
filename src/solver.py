class Solver:
    def __init__(self, model, loss_fn, optim):
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

    def fit(self, data, n_epochs):
        for i in range(1, n_epochs + 1):
            total_loss = 0
            for x, y in data:
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            print("{} loss:{}".format(i, total_loss / len(data)))
