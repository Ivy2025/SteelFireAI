import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = "results/models"
FEATURE_PATH = "models/feature_names.pkl"  # optional
DATA_CANDIDATES = [
    "data/processed_dataset.csv",
    "data/processed_dataset_planA.csv",
    "data/steel_v2_cleaned_r2.csv",
]
TARGETS_TO_PREDICT = [
    "fy_reduction",
    "fu_reduction",
    "E_reduction",
]

# Prefer plan3 models first, then fallback to planA.
PREFERRED_PLAN = "plan3"


class SteelPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Steel HT Reduction Predictor")

        self.models = {}  # {target: estimator}
        self.feature_names = None
        self.entries = {}
        self.temperature_feature = None
        self.temp_entry = None
        self.last_pred_df = None

        tk.Button(master, text="重新加载模型", command=self._load_all_models).grid(row=0, column=0, columnspan=2, pady=6)
        tk.Button(master, text="预测", command=self.predict).grid(row=1, column=0, columnspan=2, pady=6)
        tk.Button(master, text="导出预测结果 CSV", command=self.export_predictions_csv).grid(row=2, column=0, columnspan=2, pady=6)
        tk.Button(master, text="生成折线图 (20, 100-900)", command=self.plot_curves).grid(row=3, column=0, columnspan=2, pady=6)

        tk.Label(master, text="预测温度(可填多个，逗号分隔)").grid(row=4, column=0, sticky="w")
        self.temp_entry = tk.Entry(master, width=24)
        self.temp_entry.grid(row=4, column=1, sticky="w")
        self.temp_entry.insert(0, "20,100,200")

        self.result_text = tk.Text(master, height=8, width=56, state="disabled")
        self.result_text.grid(row=200, column=0, columnspan=2, pady=10)

        self._maybe_load_features()
        self._load_all_models()

    def _model_candidates(self, target):
        return [
            os.path.join(MODELS_DIR, f"catboost_{PREFERRED_PLAN}_{target}.pkl"),
            os.path.join(MODELS_DIR, f"catboost_planA_{target}.pkl"),
            os.path.join(MODELS_DIR, f"{target}_model.pkl"),  # backward compatibility
        ]

    def _load_all_models(self):
        """Load target models in fixed output order."""

        self.models.clear()
        if not os.path.isdir(MODELS_DIR):
            messagebox.showwarning("模型目录不存在", f"未找到目录: {MODELS_DIR}")
            return

        missing = []
        for target in TARGETS_TO_PREDICT:
            loaded = False
            for path in self._model_candidates(target):
                if not os.path.isfile(path):
                    continue
                try:
                    m = joblib.load(path)
                    self.models[target] = m
                    loaded = True
                    if self.feature_names is None and hasattr(m, "feature_names_"):
                        self.feature_names = list(m.feature_names_)
                    break
                except Exception:
                    continue
            if not loaded:
                missing.append(target)

        if missing:
            messagebox.showwarning(
                "部分模型未加载",
                "以下目标模型缺失或加载失败:\n" + "\n".join(missing),
            )

        if self.feature_names is not None:
            self._build_entries()

    def _build_entries(self):
        for widget in self.entries.values():
            widget.destroy()
        self.entries.clear()

        if not self.feature_names:
            return

        self.temperature_feature = self._find_temperature_feature()
        if self.temperature_feature is None:
            messagebox.showwarning("温度特征未识别", "未在特征中找到温度列，无法执行温度区间预测。")
            return

        row = 5
        for feat in self.feature_names:
            if feat in ("steel_ID", "source_ref", "diff_steel", "sample_id", self.temperature_feature):
                continue
            tk.Label(self.master, text=feat).grid(row=row, column=0, sticky="w")
            entry = tk.Entry(self.master, width=12)
            entry.grid(row=row, column=1, sticky="w")
            entry.insert(0, "0")
            self.entries[feat] = entry
            row += 1

    def _find_temperature_feature(self):
        if not self.feature_names:
            return None

        preferred = {"temperature", "temp", "test_temperature"}
        for feat in self.feature_names:
            if str(feat).lower() in preferred:
                return feat

        for feat in self.feature_names:
            if "temp" in str(feat).lower():
                return feat

        return None

    def _base_input(self):
        try:
            return {feat: float(self.entries[feat].get()) for feat in self.entries}
        except ValueError:
            return None

    def _predict_rows(self, temps):
        base_input = self._base_input()
        if base_input is None:
            messagebox.showerror("输入错误", "请确认所有特征值都是数值")
            return None

        rows = []
        for temp in temps:
            row_input = {}
            for feat in self.feature_names:
                if feat == self.temperature_feature:
                    row_input[feat] = float(temp)
                else:
                    row_input[feat] = float(base_input.get(feat, 0.0))

            X_input = pd.DataFrame([row_input], columns=self.feature_names)
            row_result = {"temperature": float(temp)}
            for target in TARGETS_TO_PREDICT:
                model = self.models.get(target)
                if model is None:
                    row_result[target] = "model not loaded"
                    continue
                try:
                    y_pred = model.predict(X_input)
                    val = y_pred[0] if hasattr(y_pred, "__iter__") else y_pred
                    row_result[target] = float(val)
                except Exception as e:
                    row_result[target] = f"error: {e}"
            rows.append(row_result)

        return rows

    def plot_curves(self):
        if not self.models or self.feature_names is None:
            messagebox.showwarning("模型未就绪", "请确保模型已放在 results/models 目录并重新加载。")
            return
        if self.temperature_feature is None:
            messagebox.showwarning("温度特征未识别", "当前模型特征中未识别到温度列。")
            return

        temps = [20.0] + [float(t) for t in range(100, 1000, 100)]
        rows = self._predict_rows(temps)
        if not rows:
            return

        df = pd.DataFrame(rows)
        if df.empty or "temperature" not in df.columns:
            messagebox.showerror("绘图失败", "未生成有效预测数据。")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Steel HT Reduction Prediction", fontsize=16)

        colors = ["b", "g", "r"]
        labels = ["fy_reduction", "fu_reduction", "E_reduction"]

        ax = axes[0, 0]
        for i, target in enumerate(labels):
            if target in df.columns:
                ax.plot(df["temperature"], df[target], marker="o", label=target, color=colors[i])
        ax.set_title("All Reductions")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Reduction Factor")
        ax.legend()
        ax.grid(True)

        plot_positions = [(0, 1), (1, 0), (1, 1)]
        for i, (target, pos) in enumerate(zip(labels, plot_positions)):
            ax = axes[pos[0], pos[1]]
            if target in df.columns:
                ax.plot(df["temperature"], df[target], marker="o", color=colors[i])
            ax.set_title(target)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Reduction Factor")
            ax.grid(True)

        plt.tight_layout()

    def predict(self):
        if not self.models or self.feature_names is None:
            messagebox.showwarning("模型未就绪", "请确保模型已放在 results/models 目录并重新加载。")
            return
        if self.temperature_feature is None:
            messagebox.showwarning("温度特征未识别", "当前模型特征中未识别到温度列。")
            return

        temps = self._parse_temperatures(self.temp_entry.get() if self.temp_entry is not None else "")
        if not temps:
            messagebox.showerror("温度输入错误", "请至少输入一个温度值，例如: 20 或 20,100,300")
            return

        rows = self._predict_rows(temps)
        if not rows:
            return

        pred_df = pd.DataFrame(rows, columns=["temperature"] + TARGETS_TO_PREDICT)
        self.last_pred_df = pred_df.copy()

        text = f"温度预测结果 ({', '.join(str(t) for t in temps)}):\n"
        text += pred_df.to_string(index=False, float_format=lambda x: f"{x:.4f}")

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state="disabled")

    def _parse_temperatures(self, raw_text):
        if raw_text is None:
            return []
        normalized = raw_text.replace("，", ",").replace(";", ",")
        parts = [p.strip() for p in normalized.split(",") if p.strip()]
        temps = []
        for p in parts:
            try:
                temps.append(float(p))
            except ValueError:
                return []
        return temps

    def export_predictions_csv(self):
        if self.last_pred_df is None or self.last_pred_df.empty:
            messagebox.showwarning("无可导出数据", "请先执行预测，再导出 CSV。")
            return

        save_path = filedialog.asksaveasfilename(
            title="保存预测结果 CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="temperature_predictions.csv",
        )
        if not save_path:
            return

        try:
            self.last_pred_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("导出成功", f"预测结果已保存到:\n{save_path}")
        except Exception as e:
            messagebox.showerror("导出失败", f"保存 CSV 时发生错误:\n{e}")

    def _maybe_load_features(self):
        """
        Prefer FEATURE_PATH; if unavailable, infer from project constants or CSV header.
        """
        try:
            fnames = joblib.load(FEATURE_PATH)
            if fnames and fnames != self.feature_names:
                self.feature_names = list(fnames)
                self._build_entries()
                return
        except Exception:
            pass

        try:
            from data_loader import FEATURE_COLUMNS
            if FEATURE_COLUMNS:
                self.feature_names = list(FEATURE_COLUMNS)
                self._build_entries()
                return
        except Exception:
            pass

        for csv_path in DATA_CANDIDATES:
            try:
                if not os.path.isfile(csv_path):
                    continue
                df = pd.read_csv(csv_path, nrows=0)
                if "N" not in df.columns and "N_pro" in df.columns:
                    df = df.rename(columns={"N_pro": "N"})
                targets = ["fy_reduction", "fu_reduction", "E_reduction"]
                drop_cols = {"steel_ID", "source_ref", "diff_steel", "sample_id"}
                feats = [c for c in df.columns if c not in targets and c not in drop_cols]
                if feats:
                    self.feature_names = feats
                    self._build_entries()
                    return
            except Exception:
                continue


if __name__ == "__main__":
    root = tk.Tk()
    app = SteelPredictorApp(root)
    root.mainloop()
