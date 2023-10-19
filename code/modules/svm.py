from argparse import ArgumentParser
from sklearn.svm import SVC

class SvmClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super(SvmClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = SVC(kernel='linear', C=1.0)  # You can adjust kernel and C parameter as needed
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # SVMs don't directly return probabilities, so I may need to adjust this part
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Similarly, I may need to adjust this part for obtaining probabilities
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        # Again, adjust this part for probabilities
        pred = y_hat.argmax(dim=-1)
        return pred

    def configure_optimizers(self):
        # SVMs don't use traditional optimizers, so this function is not used
        return None

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_features', type=int)
        parser.add_argument('--num_classes', type=int, default=15)
        return parser
