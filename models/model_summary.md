
# Fishing Detection Model Summary

## Model Performance
- **Accuracy**: 0.86
- **Cross-validation Mean**: 0.882 ± 0.016
- **Features Used**: 15

## Top 5 Features
1. speed_min (importance: 0.112)
2. speed_consistency (importance: 0.098)
3. area_covered (importance: 0.078)
4. speed_q25 (importance: 0.077)
5. distance_from_port_min (importance: 0.072)

## Model Configuration
- Algorithm: Random Forest
- Trees: 200
- Max Depth: 10
- Min Samples Split: 10

## Validation
- Cross-validation: 5-fold GroupKFold by vessel
- Evaluation: Vessel-based splitting (no data leakage)
- Training vessels: 85
- Testing vessels: 22
