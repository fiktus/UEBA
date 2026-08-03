import pandas as pd
from PySide6.QtCore import QObject, Signal
import time
from models import UEBASequenceAutoencoder
from sniffer import Sniffer

class SnifferGui(QObject):
    flows_ready = Signal(list)
    status_update = Signal(str)
    finished = Signal()

    def __init__(self, duration):
        super().__init__()
        self.duration = duration
        self.sniffer = Sniffer()
        self.flag = False

    def run(self):
        self.sniffer.run()
        start_time = time.time()
        while time.time() - start_time < self.duration and not self.flag:
            time.sleep(10)
            flows = self.sniffer.export()
            if flows:
                self.flows_ready.emit(flows)
            self.status_update.emit(f' осталось {(self.duration - time.time() + start_time)/3600:.2f} часа')
        last_flow = self.sniffer.export()
        if last_flow:
            self.flows_ready.emit(last_flow)
        self.sniffer.stop()
        self.finished.emit()

    def stop(self):
        self.flag = True

class UEBAgui(QObject):
    progress_update = Signal(int, float, float, float)
    threshold = Signal(float)
    finished = Signal(object)
    train_params_cb = Signal(int, int, int, int, int, str, int)

    def __init__(self, data_path, save_path, params):
        super().__init__()
        self.data_path = data_path
        self.params = params
        self.save_path = save_path

    def run(self):
        dataframe = pd.read_csv(self.data_path)
        model = UEBASequenceAutoencoder(**self.params)

        def train_params_cb(n_features, window_size, stride, hidden_dim, latent_dim, device, cleaned_str):
            self.train_params_cb.emit(n_features, window_size, stride, hidden_dim, latent_dim, device, cleaned_str)

        def cb(epoch, train_loss, test_loss, best_val_loss):
            self.progress_update.emit(epoch, train_loss, test_loss, best_val_loss)

        def cb_threshold(threshold):
            self.threshold.emit(threshold)

        results = model.train(dataframe, progress_callback = cb, callback_threshold = cb_threshold, train_params_cb = train_params_cb)
        model.save(self.save_path)
        self.finished.emit(model)

class MonitoringGui(QObject):
    flows_ready = Signal(list)
    results_ready = Signal(list)
    windows_ready = Signal(list)
    status_update = Signal(str)
    finished = Signal()
    header = ['timestamp', 'src', 'dst', 'proto', 'src_port', 'dst_port', 'duration', 'packets', 'bytes','tcp_flags']

    def __init__(self, model_path, max_buffer = 5000):
        super().__init__()
        self.sniffer = Sniffer()
        self.flag = False
        self.max_buffer = max_buffer
        self.model_path = model_path
        self.model = None
        self.buffer = pd.DataFrame(columns = self.header)

    def run(self):
        self.sniffer.run()
        self.model = UEBASequenceAutoencoder()
        self.model.load_model(self.model_path)
        self.status_update.emit(f'works')

        while not self.flag:
            time.sleep(10)
            flows = self.sniffer.export()
            if flows:
                self.flows_ready.emit(flows)
                result = pd.DataFrame(flows, columns = self.header)
                self.buffer = pd.concat([self.buffer, result], ignore_index = True)
                if len(self.buffer) > self.max_buffer:
                    self.buffer = self.buffer.iloc[-self.max_buffer:].reset_index(drop = True)
                mse, is_anom, status = self.model.predict(self.buffer)
                n_new = len(result)
                result['mse'] = mse[-n_new:]
                result['is_anom'] = is_anom[-n_new:]
                result['status'] = status[-n_new:]
                self.results_ready.emit(result.values.tolist())

                cleaned = self.model.process_input(self.buffer)
                win_res = self.model.score(cleaned, clean=False)
                first_indices = [idx[0] for idx in win_res['row_indices']]
                last_indices = [idx[-1] for idx in win_res['row_indices']]
                win_res['start'] = pd.to_datetime(cleaned.loc[first_indices, self.model.time_col].values, unit='s')
                win_res['end'] = pd.to_datetime(cleaned.loc[last_indices, self.model.time_col].values, unit='s')

                self.windows_ready.emit(
                    win_res[['entity', 'start', 'end', 'reconstruction_error', 'is_anomaly']].values.tolist()
                )

        self.sniffer.stop()
        self.finished.emit()

    def stop(self):
        self.status_update.emit(f'stoped')
        self.flag = True