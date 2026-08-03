import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.window_size = window_size
        self.encoder1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.15)
        self.encoder2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        #self.dropout2 = nn.Dropout(p=0.05)#может оставить##########

        self.decoder1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.dropout3 = nn.Dropout(p=0.15)
        self.decoder2 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.dropout4 = nn.Dropout(p=0.15)

        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        lengths_cpu = lengths.clamp(min=1).cpu()

        packed_in = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_enc1, _ = self.encoder1(packed_in)
        enc1_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_enc1, batch_first=True, total_length=self.window_size
        )
        enc1_out = self.dropout1(enc1_out)

        packed_enc1_dropped = nn.utils.rnn.pack_padded_sequence(
            enc1_out, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.encoder2(packed_enc1_dropped)
        latent = h_n[-1]

        context = latent.unsqueeze(1).repeat(1, self.window_size, 1)

        dec_out, _ = self.decoder1(context)
        dec_out = self.dropout3(dec_out)
        dec_out, _ = self.decoder2(dec_out)
        dec_out = self.dropout4(dec_out)

        return self.output_layer(dec_out)

class UEBASequenceAutoencoder:
    log_cols = ["duration", "packets", "bytes"]
    num_cols = ["src_port", "dst_port", "duration", "packets", "bytes"]
    rate = ("1s", "5s", "10s")
    syn = {"S", "SA", "2"}

    def __init__(self, entity_col="src", dst_col="dst", time_col="timestamp",
                 window_size=20, stride=2, hidden_dim=64, latent_dim=16,
                 device=None):
        self.entity_col = entity_col
        self.dst_col = dst_col
        self.time_col = time_col
        self.window_size = window_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = None
        self.proto_categories = None
        self.tcp_flags_categories = None
        self.dst_freq_map = None
        self.feature_cols = None
        self.feat_col_idx = None
        self.n_features = None
        self.model = None
        self.threshold = None
        self.meta = []
        self.cleaned_str = None
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def process_input(self, df):
        df = df.copy()
        for col in self.num_cols + [self.time_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        bef = len(df)
        on_drop = self.num_cols + [self.time_col, self.entity_col, self.dst_col]
        df = df.dropna(subset=on_drop)
        self.cleaned_str = bef - len(df)#имеется ввиду число строк
        print(f"очистка: отброшено {self.cleaned_str} строк")

        df["tcp_flags"] = df["tcp_flags"].fillna("NONE").astype(str)
        df.loc[df["tcp_flags"].str.strip() == "", "tcp_flags"] = "NONE"
        df["proto"] = df["proto"].fillna("OTHER").astype(str)
        return df

    def nunique_roll(self, s, w):
        return s.rolling(w).apply(lambda x: pd.Series(x).nunique(), raw=False).astype(np.float32)

    def compute_features(self, df):#по хорошему сделать то же самое по dst
        df = df.copy()
        df["ts"] = pd.to_datetime(df[self.time_col], unit="s")
        df = df.sort_values([self.entity_col, "ts"])
        chunks = []
        for entity, g in df.groupby(self.entity_col):
            orig_index = g.index
            g = g.set_index("ts")
            is_syn = g["tcp_flags"].isin(self.syn).astype(float)
            dst_codes = pd.Series(pd.factorize(g[self.dst_col])[0].astype(float), index=g.index)

            for w in self.rate:
                g[f"rate_{w}"] = g["bytes"].rolling(w).count().astype(np.float32)
                g[f"uniq_dst_{w}"] = self.nunique_roll(dst_codes, w)
                g[f"uniq_dport_{w}"] = self.nunique_roll(g["dst_port"], w)
                g[f"syn_ratio_{w}"] = is_syn.rolling(w).mean().astype(np.float32)

            g.index = orig_index
            chunks.append(g)

        return pd.concat(chunks).sort_index()

    def create_col_names(self):
        return [f"{p}_{w}" for w in self.rate for p in ("rate", "uniq_dst", "uniq_dport", "syn_ratio")]

    def fit_encoders(self, df_train):
        self.proto_categories = sorted(df_train["proto"].unique().tolist())
        self.tcp_flags_categories = sorted(df_train["tcp_flags"].unique().tolist())
        self.dst_freq_map = df_train[self.dst_col].astype(str).value_counts(normalize=True).to_dict()

    def encode_features(self, df):
        feats = pd.DataFrame(index=df.index)
        feats["src_port"] = df["src_port"].astype(np.float32)
        feats["dst_port"] = df["dst_port"].astype(np.float32)
        for col in self.log_cols:
            feats[col] = np.log1p(df[col].astype(np.float32).clip(lower=0))########################################
        feats["dst_freq"] = df[self.dst_col].astype(str).map(self.dst_freq_map).fillna(0.0).astype(np.float32)
        col_names = self.create_col_names()
        for col in col_names:
            feats[col] = df[col].astype(np.float32).fillna(0.0)

        continuous_cols = (["src_port", "dst_port"] + self.log_cols + ["dst_freq"] + col_names)

        categorical_cols = []
        for prefix, cats, series in ( ("proto", self.proto_categories, df["proto"]),
                                      ("flags", self.tcp_flags_categories, df["tcp_flags"])
                                    ):
            for cat in cats:
                feats[f"{prefix}_{cat}"] = (series == cat).astype(np.float32)
                categorical_cols.append(f"{prefix}_{cat}")

        if self.feature_cols is None:
            self.feature_cols = continuous_cols + categorical_cols
            self.feat_col_idx = list(range(len(continuous_cols)))

        return feats[self.feature_cols]


    def build_windows(self, df):
        df = self.compute_features(df)
        feats = self.encode_features(df)
        feats[self.entity_col] = df[self.entity_col].values
        feats[self.time_col] = df[self.time_col].values
        self.n_features = len(self.feature_cols)

        X_windows, meta = [], []
        for entity_id, g in feats.groupby(self.entity_col):
            g = g.sort_values(self.time_col)
            values = g[self.feature_cols].to_numpy(dtype=np.float32)
            idx = g.index.to_numpy()
            n = len(values)

            if n < self.window_size:
                pad_len = self.window_size - n
                window_values = np.concatenate(
                    [values, np.repeat(values[-1:], pad_len, axis=0)], axis=0
                )
                X_windows.append(window_values)
                meta.append((entity_id, tuple(idx)))#src и индексы реальных строк
                continue

            starts = list(range(0, n - self.window_size + 1, self.stride))#########
            if starts[-1] != n - self.window_size:#########
                starts.append(n - self.window_size)#########

            for start in starts:
                end = start + self.window_size
                X_windows.append(values[start:end])
                meta.append((entity_id, tuple(idx[start:end])))

        X = (np.array(X_windows) if X_windows else np.empty((0, self.window_size, self.n_features or 0), dtype=np.float32))###############
        self.meta = meta
        return X

    def fit_scaler(self, X_train, lens_train):
        feat_idx = self.feat_col_idx
        mask = np.arange(X_train.shape[1])[None,:] < lens_train[:,None]
        flat_cont = X_train[:, :, feat_idx][mask]
        self.scaler = StandardScaler().fit(flat_cont)

    def apply_scaler(self, X):
        X_out = X.copy()
        n, w, f = X.shape
        feat_idx = self.feat_col_idx
        flat_cont = X_out[:,:, self.feat_col_idx].reshape(-1, len(feat_idx))
        X_out[:,:, feat_idx] = self.scaler.transform(flat_cont).reshape(n, w, len(feat_idx))
        return X_out

    def valid_lens(self):
        return np.array([len(row_indices) for _, row_indices in self.meta], dtype=np.int64)

    def masked_mse_loss(self,output, target, lens):
        T, F = output.size(1), output.size(2)
        mask = (torch.arange(T, device=output.device)[None, :] < lens[:, None]).float().unsqueeze(-1)
        sq_err = (output - target) ** 2 * mask
        return sq_err.sum() / (mask.sum() * F).clamp(min=1)

    def train(self, df, val_split=0.2, epochs=40, batch_size=256, lr=3e-4, progress_callback=None, callback_threshold = None, train_params_cb = None):
        df = self.process_input(df)
        entities = df[self.entity_col].astype(str).unique()
        train_entities, val_entities = train_test_split(entities, test_size=val_split)

        self.fit_encoders(df[df[self.entity_col].astype(str).isin(train_entities)])
        X = self.build_windows(df)
        meta_entities = np.array([str(entity) for entity, _ in self.meta])
        train_idx = np.where(np.isin(meta_entities, list(train_entities)))[0]
        val_idx = np.where(np.isin(meta_entities, list(val_entities)))[0]

        valid_lens_all = self.valid_lens()
        self.fit_scaler(X[train_idx], valid_lens_all[train_idx])
        X_scaled = self.apply_scaler(X)
        print('параметры обучения: ',self.n_features, self.window_size, self.stride, self.hidden_dim, self.latent_dim, self.device)

        if train_params_cb:
            train_params_cb(self.n_features, self.window_size, self.stride, self.hidden_dim, self.latent_dim, self.device, self.cleaned_str)###

        self.model = LSTMAutoencoder(self.n_features, self.window_size, self.hidden_dim, self.latent_dim).to(self.device)


        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_scaled[train_idx]),
                torch.from_numpy(valid_lens_all[train_idx]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        val_lens = torch.from_numpy(valid_lens_all[val_idx]).to(self.device)

        val_arr = torch.from_numpy(X_scaled[val_idx]).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None

        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss_sum, n_batches = 0.0, 0
            for batch_x, batch_lens in train_loader:#Передавать в тренировочный цикл valid_lens и при расчёте MSE умножать ошибку на маску, чтобы градиент не распространялся через паддинг.
                batch_x, batch_lens = batch_x.to(self.device), batch_lens.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x, lengths=batch_lens)
                loss = self.masked_mse_loss(output, batch_x, batch_lens)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                train_loss_sum += loss.item()
                n_batches += 1

            avg_train_loss = train_loss_sum / max(1, n_batches)

            self.model.eval()

            with torch.no_grad(): #наверное вернуть батчи
                val_output = self.model(val_arr, lengths=val_lens)
                val_loss = self.masked_mse_loss(val_output, val_arr, val_lens).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())

            print(f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}")

            if progress_callback:
                progress_callback(epoch, avg_train_loss, val_loss, best_val_loss)

        self.model.load_state_dict(best_state)

        errors = self.reconstruction_error(X_scaled)
        self.threshold = float(np.quantile(errors[train_idx], 0.995))
        print(f"threshold: {self.threshold:.4f}")
        if callback_threshold:
            callback_threshold(self.threshold)
        results = pd.DataFrame(self.meta, columns=["entity", "row_indices"])
        results["reconstruction_error"] = errors
        results["is_anomaly"] = results["reconstruction_error"] > self.threshold
        return results

    def score(self, df, clean = True):
        df = self.process_input(df) if clean else df
        X = self.build_windows(df)
        errors = self.reconstruction_error(self.apply_scaler(X))
        results = pd.DataFrame(self.meta, columns=["entity", "row_indices"])
        results["reconstruction_error"] = errors
        results["is_anomaly"] = results["reconstruction_error"] > self.threshold
        return results

    def reconstruction_error(self, X_scaled, batch_size = 256, ):
        self.model.eval()
        valid_lens = self.valid_lens()
        n = len(X_scaled)
        errors = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_x = torch.from_numpy(X_scaled[start:end]).to(self.device)
                batch_lens = torch.from_numpy(valid_lens[start:end]).to(self.device)
                output = self.model(batch_x, lengths=batch_lens)
                sq_err = (output - batch_x) ** 2# чтобы получить трехмерный тензор
                batch_valid = valid_lens[start:end]
                for i, vlen in enumerate(batch_valid):
                    errors[start + i] = sq_err[i, :vlen, :].mean().item()
        return errors

    def predict(self, df):
        df = df.reset_index(drop=True)#########
        n_total = len(df)
        cleaned = self.process_input(df)
        scores = self.score(cleaned, clean = False)

        mse_scores = pd.Series(-np.inf, index=cleaned.index, dtype=np.float64)
        model_flags = pd.Series(False, index=cleaned.index, dtype=bool)
        for _, row in scores.iterrows():
            idx = list(row["row_indices"])
            mse_scores.loc[idx] = np.maximum(mse_scores.loc[idx], row["reconstruction_error"])
            model_flags.loc[idx] = model_flags.loc[idx].to_numpy() | row["is_anomaly"]

        mse_out = np.empty(n_total, dtype=np.float64)
        mse_out[:] = np.nan
        flag_out = np.zeros(n_total, dtype=bool)
        mse_out[cleaned.index] = mse_scores
        flag_out[cleaned.index] = model_flags
        status = np.where(np.isnan(mse_out), "ошибка", np.where(flag_out, "аномалия", "норма"))
        return mse_out, flag_out, status


    def save(self, path):
        checkpoint = self.__dict__.copy()
        checkpoint.pop("model", None)
        checkpoint.pop("device", None)
        checkpoint["model_state_dict"] = self.model.state_dict()
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_state = checkpoint.pop("model_state_dict")
        self.__dict__.update(checkpoint)
        self.model = LSTMAutoencoder(
            self.n_features, self.window_size, self.hidden_dim, self.latent_dim
        ).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()