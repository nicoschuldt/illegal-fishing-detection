# Illegal Fishing Detection: Scientific Findings Summary

## Project Overview

We built a machine learning system to detect illegal fishing activity using vessel movement data (AIS). This document summarizes our scientific findings across three analysis phases.

## Dataset

- **Source**: Global Fishing Watch labeled AIS data
- **Vessels**: 110 longline fishing vessels  
- **Data Points**: 13.9 million AIS positions
- **Labeled Points**: 219,741 with fishing/non-fishing labels
- **Time Period**: 2012-2014

## Phase 1: Initial Analysis (Notebook 01)

### Problem: Data Leakage Discovery

**What we found**: Initial random splitting gave unrealistic 99% accuracy.

**Why this happened**: 
- Random splitting put points from the same vessel in both training and test sets
- Model learned vessel-specific patterns instead of general fishing behavior
- This is called "data leakage" - a common mistake in time series analysis

**Solution**: Vessel-based splitting
- Entire vessels go into either training OR testing, never both
- This tests if the model works on completely new vessels

**Result**: Accuracy dropped to 77% - realistic but much lower performance.

### Key Learning
Proper evaluation methodology is critical. The 77% baseline represents genuine model performance on unseen vessels.

## Phase 2: Feature Engineering (Notebook 02)

### Problem: Improve Model Performance

**Approach**: Engineer behavioral features that capture fishing patterns.

**Features developed** (52 total features):
- **Speed patterns**: Average speed, speed variability, minimum speed
- **Movement patterns**: Course changes, turn frequency, path efficiency  
- **Spatial context**: Distance from shore/port, area covered
- **Temporal patterns**: Time of day, trip duration

### Model Training

**Feature selection**: Reduced to 15 most important features to prevent overfitting.

**Top 3 features**:
1. `speed_min` - Fishing vessels slow down or stop
2. `course_changes_per_hour` - Fishing involves more maneuvering
3. `speed_q25` - Low-speed behavior patterns

**Validation**: 5-fold cross-validation using vessel grouping.

### Results

- **Accuracy**: 88.19% ± 3.1% (cross-validated)
- **F1 Score**: 91.82% ± 2.1%
- **Improvement**: +11 percentage points over baseline

### Key Learning
Behavioral features significantly improve fishing detection. The model now generalizes well to new vessels.

## Phase 3: Unsupervised Learning (Notebook 03)

### Question: Can clustering reveal natural vessel behavior patterns?

**Approach**: Test if vessels naturally group into behavioral categories (e.g., "fishing boats" vs "transit boats").

### Methods tested:
- **K-means clustering**: Groups vessels by similarity
- **DBSCAN**: Finds density-based clusters  
- **Hierarchical clustering**: Creates nested groupings

### Results

**Optimal clustering**: k=2 clusters (statistically validated)
- **Cluster 0**: 67.2% fishing rate (69% of trips)
- **Cluster 1**: 74.6% fishing rate (31% of trips)

**Cluster quality**: Silhouette score = 0.213 (weak separation)

**DBSCAN result**: All data classified as "noise" - no natural clusters exist

**Impact on supervised learning**: Clustering decreased accuracy by -1.1%

### Key Learning
Vessel behavior exists on a **continuous spectrum** rather than discrete categories. Clustering doesn't improve fishing detection.

## Scientific Conclusions

### 1. Methodology Validation
- Vessel-based evaluation prevents data leakage
- Cross-validation with vessel grouping gives realistic performance estimates
- Proper evaluation dropped accuracy from 99% to 77%, then feature engineering improved it to 88%

### 2. Feature Engineering Success  
- Behavioral features (speed, turns, movement patterns) are highly predictive
- 15 carefully selected features capture all relevant information
- Model accuracy is reliable and well-calibrated

### 3. Supervised Learning is Optimal
- No natural vessel behavior categories exist
- Fishing activity is best predicted directly from movement features
- Clustering approaches don't add value

### 4. Model Performance
- **88.19% accuracy** on completely unseen vessels
- **High precision and recall** for fishing detection
- **Well-calibrated probabilities** (when model says 80% fishing, it's fishing ~80% of the time)

## Business Impact

### Enforcement Applications
- **Real-time detection**: Process 12 hours of vessel data in seconds
- **Risk scoring**: Continuous 0-100% fishing probability scores
- **MPA violations**: Combine fishing predictions with protected area boundaries
- **Inspection prioritization**: Focus on vessels with high fishing scores in restricted areas

### Te