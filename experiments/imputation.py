import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import gpt_imp, bert_imp, transformer_imp, timesnet_imp, saits_imp, llama_imp, phi_imp, gemma_imp, t5_imp, bart_imp
from utils.early_stopping import EarlyStopping
from utils.utils import adjust_learning_rate, visual
from utils.metrics import metric
from data_provider.data_factory import get_dataloader_and_dataset


class Imputation:
    """
    Imputation is a class that encapsulates the imputation process for time-series data, including training, validation, and testing of imputation models.

    Attributes:
        args: A configuration object containing parameters and settings for the imputation model.
        model_dict: A dictionary containing the imputation models available for use.
        device: The device (CPU or GPU) to be used for the imputation process.
        model: The imputation model to be used for the imputation process.
    Args:
        args: A configuration object containing parameters and settings for the imputation model.
        **kwargs: Additional keyword arguments to be passed to the model.
    """
    def __init__(self, args, **kwargs):
        self.args = args
        self.model_dict = {
            "BartImputer": bart_imp,
            "BertImputer": bert_imp,
            "GemmaImputer": gemma_imp,
            "GptImputer": gpt_imp,
            "LlamaImputer": llama_imp,
            "PhiImputer": phi_imp,
            "SaitsImputer": saits_imp,
            "T5Imputer": t5_imp,
            "TimesNetImputer": timesnet_imp,
            "TransformerImputer": transformer_imp,
        }
        self.device = self._acquire_device()
        self.model = self._build_model(**kwargs).to(self.device)

    def _build_model(self, **kwargs):
        model = None
        if self.args.model == "BartImputer":
            model = self.model_dict[self.args.model].BartImputer(self.args).float()
        elif self.args.model == "BertImputer":
            model = self.model_dict[self.args.model].BertImputer(self.args).float()
        elif self.args.model == "GemmaImputer":
            model = self.model_dict[self.args.model].GemmaImputer(self.args).float()
        elif self.args.model == "GptImputer":
            model = self.model_dict[self.args.model].GptImputer(self.args).float()
        elif self.args.model == "LlamaImputer":
            model = self.model_dict[self.args.model].LlamaImputer(self.args).float()
        elif self.args.model == "PhiImputer":
            model = self.model_dict[self.args.model].PhiImputer(self.args).float()
        elif self.args.model == "SaitsImputer":
            model = self.model_dict[self.args.model].SaitsImputer(self.args, **kwargs).float()
        elif self.args.model == "T5Imputer":
            model = self.model_dict[self.args.model].T5Imputer(self.args).float()
        elif self.args.model == "TimesNetImputer":
            model = self.model_dict[self.args.model].TimesNetImputer(self.args).float()
        elif self.args.model == "TransformerImputer":
            model = self.model_dict[self.args.model].TransformerImputer(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = get_dataloader_and_dataset(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0)
        return model_optimizer

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                print("Neither CUDA nor MPS device found.")
                device = None
        else:
            device = None  # CPU
        return device

    def validate(self, validate_data, validate_loader, criterion):
        """
        Validate the model on the validation set.

        Args:
            validate_data: The validation dataset.
            validate_loader: The validation data loader.
            criterion: The loss function to be used for validation.

        Returns:
            total_loss: The total loss on the validation set.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(validate_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, mask)

                # f_dim = -1 if self.args.features == 'MS' else 0
                f_dim = 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        Trains a machine learning model using specified training settings, managing data loading,
        training iterations, validation, and early stopping based on validation loss.

        This method handles saving model parameters, training the model with batch-wise
        gradient updates, evaluating it on validation and test sets, and implementing early stopping
        to prevent overfitting. The method supports random masking of input data as part of the
        training process. At the end of training, the model with the best performance on the
        validation set is saved and loaded as the final model state.

        Args:
            setting (str): A string that specifies the training configuration and is used to
                           create directories and file names for saving model parameters and losses.

        Returns:
            torch.nn.Module: The trained model with weights loaded from the best-performing
                             checkpoint during the training process.

        Side Effects:
            - Saves model parameters and loss statistics to disk.
            - Prints training progress and loss metrics to standard output.
            - Modifies internal states of the model and optimizer used in training.

        Raises:
            FileNotFoundError: If the checkpoint file for the best model is not found.
            Exception: If there are issues during the file I/O operations or during model training.

        This will start the training process with settings defined in 'experiment_1', and
        return the trained model after potentially early stopping if specified in the training arguments.
        """
        # save trainable params (just for llms)
        if 'llm_imp' in str(type(self.model).__mro__):
            folder_path_params = './results/' + self.args.run_name + '/' + setting
            if not os.path.exists(folder_path_params):
                os.makedirs(folder_path_params)
            df_params = pd.DataFrame({'trainable_params': [self.model.trainable_params],
                                      'all_param': [self.model.all_param]})

            df_params.to_csv(path_or_buf=os.path.join(folder_path_params,
                                                      self.args.model + '_' +
                                                      self.args.data + '_' +
                                                      str(self.args.mask_rate) + '_' + 'trainable_params' +
                                                      '.csv'))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='validation')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.run_name, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_epochs = []
        vali_loss_epochs = []
        test_loss_epochs = []

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optimizer.zero_grad()  # Zero the gradients for every batch

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, mask)

                f_dim = 0  # -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optimizer.step()  # Adjust learning weights

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.validate(vali_data, vali_loader, criterion)
            test_loss = self.validate(test_data, test_loader, criterion)

            train_loss_epochs.append(train_loss)
            vali_loss_epochs.append(vali_loss)
            test_loss_epochs.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optimizer, epoch + 1, self.args)

        df_losses = pd.DataFrame({
            'Index': range(1, self.args.epochs + 1),
            'train_loss': train_loss_epochs,
            'vali_loss': vali_loss_epochs,
            'test_loss': test_loss_epochs
        })

        folder_path = './results/' + self.args.run_name + '/' + setting + '/losses/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_losses.to_csv(path_or_buf=os.path.join(folder_path,
                                                  self.args.model + '_' +
                                                  self.args.data + '_' +
                                                  str(self.args.mask_rate) + '_' +
                                                  '.csv'))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        Evaluates the trained model on a test dataset, potentially after loading a saved model state.
        This method processes the test data using the model, optionally loads model weights from a
        checkpoint, performs predictions, and calculates various evaluation metrics. It supports the
        ability to save predictions and true values for further analysis and visualizations.

        This method performs masking on the test data, similar to the training phase, to simulate
        missing data scenarios, and then imputes these missing values using the model's outputs.

        Args:
            setting (str): A string that specifies the directory suffix where model checkpoints and
                           results are stored and retrieved.
            test (int, optional): A flag (0 or 1) indicating whether to load the model from a saved
                                  checkpoint. Default is 0, which means the model is used in its current
                                  state. If 1, the model is loaded from a checkpoint specified by the 'setting'.

        Side Effects:
            - Loads the model state from disk if 'test' is set to 1.
            - Saves evaluation metrics and optionally the prediction results to the disk.
            - Prints the shapes of the prediction and true arrays to standard output.
            - Outputs evaluation metrics to a result file.

        This sequence trains the model with the configuration 'experiment_1', and then tests it,
        loading the best model state from the training phase.
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.run_name + '/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './results/' + self.args.run_name + '/' + setting + '/ts_plot_data/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, mask)

                # eval
                # f_dim = -1 if self.args.features == 'MS' else 0
                f_dim = 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                mask_for_eval = mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 1000 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    df = pd.DataFrame({
                        'Index': range(1, len(true[0, :, -1]) + 1),
                        'true': true[0, :, -1],
                        'preds': filled,
                        'mask': mask_for_eval[0, :, -1]
                    })

                    df.to_csv(path_or_buf=os.path.join(folder_path,
                                                       self.args.model + '_' +
                                                       self.args.data + '_' +
                                                       str(self.args.mask_rate) + '_' +
                                                       str(i) + '.csv'))
                    # visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + self.args.run_name + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return
