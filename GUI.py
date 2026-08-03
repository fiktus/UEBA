import csv
import os
import sys
from datetime import datetime
from PySide6 import QtWidgets
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QListWidgetItem, QListWidget, \
    QStackedWidget, QVBoxLayout, QTextEdit, QTableWidget, QTableWidgetItem, QProgressBar, QFileDialog, QTabWidget
from connect import SnifferGui
from connect import UEBAgui, MonitoringGui
from PySide6.QtGui import QColor
#later в сыром выводе выводить всю таблицу с созданными фичами
#later добавить в мониторинг маленького агента для анализа строк
class MonitoringPage(QWidget):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.thread = None
        self.worker = None

        layout = QVBoxLayout(self)
        self.log = QTextEdit()
        self.log.setMaximumHeight(100)
        self.table = QTableWidget()
        self.entity_header = ['entity', 'windows', 'anomalies', 'anomaly_ratio']
        self.header = ['timestamp', 'src', 'dst', 'proto', 'src_port', 'dst_port', 'duration', 'packets', 'bytes', 'tcp_flags'] + ['mse', 'is_anom', 'status']
        self.window_header = ['entity', 'window_start', 'window_end', 'mse', 'is_anomaly']

        self.window_table = QTableWidget()
        self.window_table.setColumnCount(len(self.window_header))
        self.window_table.setHorizontalHeaderLabels(self.window_header)
        self.window_table.horizontalHeader().setStretchLastSection(True)
        self.window_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.entity_table = QTableWidget()
        self.entity_table.setColumnCount(len(self.entity_header))
        self.entity_table.setHorizontalHeaderLabels(self.entity_header)
        self.entity_table.horizontalHeader().setStretchLastSection(True)
        self.entity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.entity_stats = {}
        self.seen_windows = set()

        self.table.setColumnCount(len(self.header))
        self.table.setHorizontalHeaderLabels(self.header)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.entity_table, 'entitys')
        self.tabs.addTab(self.window_table, 'windows')
        self.tabs.addTab(self.table, 'raw output')

        self.load_model_btn = QPushButton('Load model')
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')
        self.start_button.setEnabled(False)

        layout.addWidget(self.log)
        layout.addWidget(self.tabs, stretch=1)
        layout.addWidget(self.load_model_btn)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.load_model_btn.clicked.connect(self.load_model)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.finish)

        self.thread = None
        self.sniffer = None

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self,'choose model', '', '(*pt)')
        if path:
            self.model_path = path
            self.log.append(f'chosen model path: {self.model_path}')
            self.start_button.setEnabled(True)

    def start(self):
        self.start_button.setEnabled(False)
        self.table.setRowCount(0)
        self.thread = QThread()
        self.worker = MonitoringGui( model_path= self.model_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.results_ready.connect(self.add_rows)
        self.worker.windows_ready.connect(self.add_window_rows)
        self.worker.status_update.connect(self.log.append)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def finish(self):
        if self.worker:
            self.worker.stop()
        self.log.append('stopped')

    def add_rows(self, flows):
        anom_idx = self.header.index('is_anom')
        for flow in flows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            is_anom = bool(flow[anom_idx])
            for e, i in enumerate(flow):
                item = QTableWidgetItem(str(i))
                if is_anom:
                    item.setBackground(QColor('red'))
                self.table.setItem(row, e, item)

    def add_window_rows(self, windows):#####
        self.window_table.setRowCount(0)
        for w in windows:
            entity, start, end, is_anom = w[0], w[1], w[2], bool(w[-1])

            row = self.window_table.rowCount()
            self.window_table.insertRow(row)
            for e, val in enumerate(w):
                item = QTableWidgetItem(str(val))
                if is_anom:
                    item.setBackground(QColor('red'))
                self.window_table.setItem(row, e, item)
            key = (entity, start, end)
            if key not in self.seen_windows:
                self.seen_windows.add(key)
                stats = self.entity_stats.setdefault(entity, [0, 0])
                stats[0] += 1
                stats[1] += is_anom
        self.refresh_entity_table()

    def refresh_entity_table(self):
        self.entity_table.setRowCount(0)
        for entity, (total, anoms) in sorted(self.entity_stats.items(), key=lambda kv: -kv[1][1]):
            row = self.entity_table.rowCount()
            self.entity_table.insertRow(row)
            ratio = anoms / total
            for e, val in enumerate([entity, total, anoms, f'{ratio:.0%}']):
                item = QTableWidgetItem(str(val))
                if ratio >= 0.3:
                    item.setBackground(QColor('red'))
                self.entity_table.setItem(row, e, item)


class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.threshold = None
        self.model_name = None
        self.thread = None
        self.data_path = None
        self.trainer = None
        self.model = None
        self.model_path = None

        layout = QVBoxLayout(self)
        self.log = QTextEdit()
        self.progress = QProgressBar()
        self.load_btn = QPushButton('Load data')
        self.save_btn = QPushButton('Save model')
        self.start_btn = QPushButton('Start training')
        self.start_btn.setEnabled(False)

        layout.addWidget(self.log)
        layout.addWidget(self.progress)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.start_btn)

        self.load_btn.clicked.connect(self.load_data)
        self.save_btn.clicked.connect(self.ch_save_path)
        self.start_btn.clicked.connect(self.start_train)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, 'choose csv file', '', 'CSV (*.csv)')
        if path:
            self.data_path = path
            self.log.append(f'chosen dataframe path: {self.data_path}')
            self.start_check()

    def ch_save_path(self):
        path = QFileDialog.getExistingDirectory(self, 'choose model save path', '')
        if path:
            now = datetime.now()
            self.model_name = f'Model_{str(now.hour)}_{str(now.minute)}_{str(now.second)}.pt'
            self.model_path = os.path.join(path, self.model_name)
            self.model_path = os.path.normpath(self.model_path)
            self.log.append(f'chosen model path: {self.model_path}')
            self.start_check()

    def start_check(self):
        if self.model_path and self.data_path:
            self.start_btn.setEnabled(True)

    def start_train(self):
        self.start_btn.setEnabled(False)
        self.progress.setRange(0, 30)
        self.thread = QThread()
        self.trainer = UEBAgui(self.data_path, self.model_path, {'window_size': 40, 'stride': 5})#window_size=20, stride=2
        #
        self.trainer.moveToThread(self.thread)

        self.thread.started.connect(self.trainer.run)
        self.trainer.progress_update.connect(self.epoch_log)
        self.trainer.threshold.connect(self.threshold_cb)
        self.trainer.train_params_cb.connect(self.train_params)
        self.trainer.finished.connect(self.on_finished)
        self.trainer.finished.connect(self.thread.quit)
        self.trainer.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def epoch_log(self, epoch, train_loss, test_loss, best_loss):
        self.progress.setValue(epoch+1)
        self.log.append(f'epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}, best_loss: {best_loss} ')

    def threshold_cb(self, threshold):
        self.threshold = threshold
        self.log.append(f'threshold: {self.threshold}')

    def train_params(self, n_features, window_size, stride, hidden_dim, latent_dim, device, cleaned_str):
        self.log.append(f'number of input features: {n_features}\n'
                        f'Time window length: {window_size}\n'
                        f'window stride: {stride}\n'
                        f'hidden dim: {hidden_dim}'
                        f'latent_dim: {latent_dim}\n'
                        f'device: {device}\n'
                        f'rows dropped: {cleaned_str}\n')

    def on_finished(self, model):
        self.model = model
        self.start_btn.setEnabled(True)
        self.log.append(f'train completed, save path: {self.model_path}')###


class CollectionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.header = ['timestamp', 'src', 'dst', 'proto', 'src_port', 'dst_port', 'duration', 'packets', 'bytes', 'tcp_flags']
        self.file = None

        layout = QVBoxLayout(self)
        self.log = QTextEdit()
        self.log.setMaximumHeight(100)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.header))
        self.table.setHorizontalHeaderLabels(self.header)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')

        layout.addWidget(self.log)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.finish)
        self.thread = None
        self.sniffer = None

    def start(self):
        self.start_button.setEnabled(False)
        now = datetime.now()
        self.file = f'flow_{str(now.hour)}_{str(now.minute)}_{str(now.second)}.csv'
        self.stop_button.setEnabled(True)
        self.thread = QThread()
        self.table.setRowCount(0)

        self.sniffer = SnifferGui(duration= 3600*24)
        self.sniffer.moveToThread(self.thread)
        self.thread.started.connect(self.sniffer.run)
        self.sniffer.flows_ready.connect(self.add_rows)
        self.sniffer.flows_ready.connect(self.append_to_csv)
        self.sniffer.status_update.connect(self.log.clear)
        self.sniffer.status_update.connect(self.log.append)

        self.sniffer.finished.connect(self.stop_sniffer)
        self.sniffer.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def finish(self):
        if self.sniffer:
            self.sniffer.stop()
        self.log.append('collection stopped')

    def stop_sniffer(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log.append('collection completed')

    def append_to_csv(self, flows):
        if not os.path.exists(self.file):
            with open(self.file, 'a', newline='', encoding= 'UTF-8') as f:
                csv.writer(f).writerow(self.header)
        with open(self.file, 'a', newline='', encoding= 'UTF-8') as f:
            csv.writer(f).writerows(flows)

    def add_rows(self, flows):
        for flow in flows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for e, i in enumerate(flow):
                item = QTableWidgetItem(str(i))
                self.table.setItem(row, e, item)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.menu = QListWidget()
        self.stack = QStackedWidget()
        self.create_window()
        self.Qstyle()

    def create_window(self):
        self.resize(1000, 600)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        self.menu.setFixedWidth(200)

        self.menu.addItem(QListWidgetItem('Monitoring'))
        self.menu.addItem(QListWidgetItem('Training'))
        self.menu.addItem(QListWidgetItem('Collection data'))

        self.stack.addWidget(MonitoringPage())
        self.stack.addWidget(TrainingPage())
        self.stack.addWidget(CollectionPage())

        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)

        main_layout.addWidget(self.menu)
        main_layout.addWidget(self.stack)
        self.menu.setCurrentRow(0)

    def Qstyle(self):
        menu_style = """
            QListWidget {
                background: #262b2b;
                color: white;
                font-size: 20px;
            }
            QListWidget::item {
                height: 50px; 
            }  
        """

        self.menu.setStyleSheet(menu_style)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())