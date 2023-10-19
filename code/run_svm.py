import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modules import FeatureDataModule, SvmClassifier  # Assuming you have an SvmClassifier class


def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('name')
    parser = FeatureDataModule.add_argparse_args(parser)
    parser = SvmClassifier.add_argparse_args(parser)  # Adjusted for SvmClassifier
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(accelerator='gpu', devices=1,
                        default_root_dir=osp.abspath(
                            osp.dirname(__file__), '../data/svm')))  # Adjusted directory
    args = parser.parse_args(argv)
    return args


def main(args):
    data_module = FeatureDataModule(args)
    model = SvmClassifier(args)  # Adjusted for SvmClassifier
    logger = TensorBoardLogger(args.default_root_dir, args.name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)
    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback])
    # SVMs don't use traditional optimizers, so the .fit() method is not used for training
    # You will need to implement your own training procedure for SVM
    # Similarly, the .predict() method for SVM may require different handling
    # Ensure that you handle training and prediction accordingly for SVM
    # Example:
    # svm_model.fit(data_module.train_dataloader())
    # predictions = svm_model.predict(data_module.test_dataloader())
    # ...
    # ...
    # ...
    # Adjust the code according to your SVM implementation

    # Example: Save SVM model
    svm_model.save_model(logger.log_dir)  

    # Example: Load SVM model
    # svm_model.load_model(ckpt_path='best')  # Provide correct path if needed

    # Example: Make predictions
    # predictions = svm_model.predict(data_module.test_dataloader())

    # Example: Save predictions to a CSV file
    # df = data_module.test_df.copy()
    # df['Category'] = predictions
    # prediction_path = osp.join(logger.log_dir, 'test_prediction.csv')
    # df.to_csv(prediction_path, index=False)
    # print('Output file:', prediction_path)


if __name__ == '__main__':
    main(parse_args())
