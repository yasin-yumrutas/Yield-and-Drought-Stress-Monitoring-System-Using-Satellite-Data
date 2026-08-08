import sys
import os
import pandas as pd
import json

from train import run_pipeline
from src.inference import CropHealthPredictor

def run_demo():
    print("\n=================================================================")
    print(" Running Satellite Crop Health & Drought Early Warning Inference ")
    print("=================================================================")
    
    if not os.path.exists('data/drought_timeseries.csv'):
        print("Dataset not found! Training pipeline starting first...")
        run_pipeline()

    df_ts = pd.read_csv('data/drought_timeseries.csv')
    predictor = CropHealthPredictor()
    
    # Pick 3 representative sample fields (Healthy, Mild Stress, Severe Stress)
    sample_fields = [1, 5, 12]
    
    reports = []
    for f_id in sample_fields:
        field_ts = df_ts[df_ts['field_id'] == f_id]
        if not field_ts.empty:
            res = predictor.predict_field_health(field_ts)
            res['field_id'] = f_id
            reports.append(res)
            
            print(f"\n--- Tarla ID: {f_id} Tahmin Raporu ---")
            print(f"Güncel NDVI (Bitki Sağlığı): {res['current_ndvi']:.2f}")
            print(f"Güncel NDWI (Su Stresi): {res['current_ndwi']:.2f}")
            print(f"14-Gün Sonrası Erken Uyarı Durumu: {res['predicted_14d_stress_status']}")
            print(f"Harita Renk Kodu: {res['map_color_code']}")
            print(f"Stres Olasılıkları -> Normal: %{res['stress_probabilities']['normal_prob']*100:.1f} | "
                  f"Orta Stres: %{res['stress_probabilities']['mild_stress_prob']*100:.1f} | "
                  f"Şiddetli Kuraklık: %{res['stress_probabilities']['severe_drought_prob']*100:.1f}")
            print(f"Tahmini Sezon Rekoltesi: {res['forecasted_yield_kg_per_da']} kg/dönüm")

    os.makedirs('reports', exist_ok=True)
    with open('reports/sample_inference_report.json', 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
        
    print("\nInference complete! JSON report exported to 'reports/sample_inference_report.json'")

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == "--gui" or sys.argv[1] == "-g"):
        from desktop_gui import main as launch_desktop_gui
        launch_desktop_gui()
    elif len(sys.argv) > 1 and sys.argv[1] == "--train":
        run_pipeline()
    else:
        # Default: Train if needed, then run demo inference
        if not os.path.exists('models/drought_xgboost.joblib'):
            run_pipeline()
        run_demo()
