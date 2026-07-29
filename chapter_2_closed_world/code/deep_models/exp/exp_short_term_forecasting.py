from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    @staticmethod
    def _stack_m4_series(series):
        return np.stack([np.asarray(ts, dtype=np.float32) for ts in series]).astype(np.float32)

    @staticmethod
    def _m4_smape_by_series(forecast, target):
        denominator = np.abs(target) + np.abs(forecast)
        denominator[denominator == 0.0] = 1.0
        return np.mean(200.0 * np.abs(forecast - target) / denominator, axis=1)

    @staticmethod
    def _m4_mase_by_series(forecast, target, insample, frequency):
        values = []
        for index in range(len(forecast)):
            history = np.asarray(insample[index], dtype=np.float64)
            scale = np.mean(np.abs(history[frequency:] - history[:-frequency]))
            values.append(np.nan if scale == 0.0 else np.mean(np.abs(forecast[index] - target[index])) / scale)
        return np.asarray(values, dtype=np.float64)

    def _save_m4_channel_metrics(self, train_data, test_data, preds, trues):
        if not self.args.channel_results_root:
            return
        ids = np.asarray(test_data.ids).astype(str)
        selected = np.asarray(test_data.reversal_selected_mask, dtype=bool)
        horizon = self.args.pred_len
        naive_frame = pandas.read_csv(os.path.join(self.args.root_path, 'submission-Naive2.csv'))
        naive_frame.iloc[:, 0] = naive_frame.iloc[:, 0].astype(str)
        naive_frame = naive_frame.set_index(naive_frame.columns[0])
        naive = naive_frame.loc[ids].iloc[:, :horizon].to_numpy(dtype=np.float64)
        if self.args.reversal_case[1] == 'R':
            naive[selected] = naive[selected, ::-1]

        forecast = preds[:, :, 0].astype(np.float64)
        target = trues.astype(np.float64)
        frequency = M4Meta.frequency_map[self.args.seasonal_patterns]
        model_smape = self._m4_smape_by_series(forecast, target)
        model_mase = self._m4_mase_by_series(
            forecast, target, train_data.timeseries, frequency)
        naive_smape = self._m4_smape_by_series(naive, target)
        naive_mase = self._m4_mase_by_series(
            naive, target, train_data.timeseries, frequency)
        with np.errstate(divide='ignore', invalid='ignore'):
            series_owa = 0.5 * (model_smape / naive_smape + model_mase / naive_mase)

        output_dir = os.path.join(
            self.args.channel_results_root, 'm4', self.args.model, self.args.reversal_case)
        os.makedirs(output_dir, exist_ok=True)
        frame = pandas.DataFrame({
            'dataset': 'M4',
            'seasonal_pattern': self.args.seasonal_patterns,
            'model': self.args.model,
            'case': self.args.reversal_case,
            'series_id': ids,
            'selected_for_reversal': selected,
            'reversed_train': selected & (self.args.reversal_case[0] == 'R'),
            'reversed_eval': selected & (self.args.reversal_case[1] == 'R'),
            'smape': model_smape,
            'mase': model_mase,
            'naive2_smape': naive_smape,
            'naive2_mase': naive_mase,
            'owa_series_ratio': series_owa,
        })
        frame.to_csv(
            os.path.join(output_dir, f'{self.args.seasonal_patterns}_channel_metrics.csv'), index=False)

        group_summary = pandas.DataFrame([{
            'group': self.args.seasonal_patterns,
            'series_count': len(frame),
            'selected_count': int(selected.sum()),
            'smape': float(np.nanmean(model_smape)),
            'mase': float(np.nanmean(model_mase)),
            'naive2_smape': float(np.nanmean(naive_smape)),
            'naive2_mase': float(np.nanmean(naive_mase)),
        }])
        group_summary['owa'] = 0.5 * (
            group_summary['smape'] / group_summary['naive2_smape']
            + group_summary['mase'] / group_summary['naive2_mase'])
        group_summary.to_csv(
            os.path.join(output_dir, f'{self.args.seasonal_patterns}_official_summary.csv'), index=False)
        self._write_m4_official_aggregate(output_dir)

    @staticmethod
    def _write_m4_official_aggregate(output_dir):
        groups = M4Meta.seasonal_patterns
        files = [os.path.join(output_dir, f'{group}_official_summary.csv') for group in groups]
        if not all(os.path.exists(path) for path in files):
            return
        detailed = pandas.concat([pandas.read_csv(path) for path in files], ignore_index=True)
        detailed = detailed.set_index('group')
        rows = []

        def aggregate(label, members):
            counts = detailed.loc[members, 'series_count'].to_numpy(dtype=np.float64)
            row = {'group': label, 'series_count': int(counts.sum())}
            for metric in ['smape', 'mase', 'naive2_smape', 'naive2_mase']:
                values = detailed.loc[members, metric].to_numpy(dtype=np.float64)
                row[metric] = float(np.sum(values * counts) / np.sum(counts))
            row['owa'] = 0.5 * (
                row['smape'] / row['naive2_smape'] + row['mase'] / row['naive2_mase'])
            return row

        for group in ['Yearly', 'Quarterly', 'Monthly']:
            rows.append(aggregate(group, [group]))
        rows.append(aggregate('Others', ['Weekly', 'Daily', 'Hourly']))
        rows.append(aggregate('Average', groups))
        pandas.DataFrame(rows).to_csv(os.path.join(output_dir, 'official_summary.csv'), index=False)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(self._stack_m4_series(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()
            id_list = np.arange(0, B, 500)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.numpy()

            preds = outputs
            trues = self._stack_m4_series(y)
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], max(1, preds.shape[0] // 10)):
                if self.args.skip_visualization:
                    break
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # result save
        m4_result_dir = getattr(self.args, 'm4_result_dir', '') or self.args.model
        folder_path = os.path.join('./m4_results', m4_result_dir) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.insert(0, 'id', test_loader.dataset.ids[:preds.shape[0]])
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv', index=False)
        self._save_m4_channel_metrics(train_loader.dataset, test_loader.dataset, preds, trues)

        print(m4_result_dir)
        file_path = folder_path
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            if self.args.reversal_mask_path:
                print('Case-aware partial-reversal M4 metrics are saved under channel_results_root.')
            else:
                m4_summary = M4Summary(file_path, self.args.root_path)
                # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
                smape_results, owa_results, mape, mase = m4_summary.evaluate()
                print('smape:', smape_results)
                print('mape:', mape)
                print('mase:', mase)
                print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
