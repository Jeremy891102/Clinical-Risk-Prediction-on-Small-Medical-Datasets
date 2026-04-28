"""Quick validation: TabPFN works on the heart dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.pipeline import run_single_experiment
from src.models.registry import get_model

model = get_model("tabpfn")
print("Running TabPFN on heart dataset (this may take 1-2 minutes on CPU)...")
result = run_single_experiment(model, "heart", seed=42)
print(f"\nAUROC: {result['roc_auc']:.3f}")
print(f"Fit time: {result.get('fit_time_sec', 'N/A')}s")
print("Expected AUROC range: 0.85 - 0.96 (other models scored 0.93-0.96 on this dataset)")
