# Initial Data Exploration and Model Validation

### Problem Statement

Can we detect when a vessel is actively fishing based on its movement patterns and other available data? This is the first step toward our ultimate goal of identifying potentially illegal fishing in protected areas.

## Data Overview

### Data Source

We're using labeled AIS (Automatic Identification System) data from Global Fishing Watch, which includes:
- Vessel identifiers (mmsi)
- Position (latitude, longitude)
- Speed and course
- Distance from shore and port
- Labels indicating whether the vessel was fishing or not

### Data Overview

- 14 million AIS data points from 110 unique fishing vessels
- 219,741 labeled points (is_fishing != -1)
- 63.79% of labeled points represent fishing activities


## Key Findings

### Data Patterns
!
- **Speed**: Fishing predominantly occurs at lower speeds (0-5 knots)
- **Location**: Fishing tends to cluster in specific areas while non-fishing forms travel paths
- **Time**: Modest variations in fishing activity throughout the day

### Model Validation Challenge

We tested two evaluation approaches:

1. **Naive Random Split**: 
   - Randomly distributing data points between training and testing sets
   - Result: 99.38% accuracy (misleadingly high)

2. **Vessel-Based Split**:
   - Ensuring all data from a vessel is either in training OR testing
   - Result: 76.96% accuracy (more realistic)

| Metric | Naive Approach | Vessel-Based Approach |
|--------|----------------|----------------------|
| Accuracy | 99.38% | 76.96% |
| F1 (Class 0) | 0.99 | 0.66 |
| F1 (Class 1) | 1.00 | 0.83 |

### Why Such a Big Difference?

The 23% drop in accuracy reveals **data leakage** in the naive approach:
- The model memorized specific vessel patterns rather than learning generalizable fishing behavior
- With random splitting, the model sees some points from each vessel in training and others in testing
- The vessel-based approach forces the model to predict on completely new vessels

### Feature Importance
- Both models identified speed as the most predictive feature
- The naive model relied heavily on location (lat/lon)
- The vessel-based model gave more weight to behavioral features

## Implications
1. **Realistic baseline**: 77% accuracy represents our true performance on unseen vessels
2. **Proper validation matters**: Especially for spatiotemporal data where dependencies exist
3. **Feature engineering direction**: Focus on vessel-agnostic behavioral features rather than location-specific patterns

Our next steps will focus on developing movement pattern features that generalize well across different vessels and fishing regions.


## What We've Learned

1. **Proper validation is crucial**: The way we split data can dramatically affect how we evaluate model performance
2. **77% accuracy is our true baseline**: This represents how well our model generalizes to new vessels
3. **Movement patterns matter**: Speed and distance features are most predictive of fishing activity
4. **Location dependency is a challenge**: Models can become too dependent on specific fishing grounds
