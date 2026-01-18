# Multi-Rep Exercise Analysis System - Complete Documentation Index

## 📋 Quick Navigation

### For First-Time Users
1. Start with: **[QUICK_START.md](QUICK_START.md)** - 5-minute overview
2. Then read: **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Architecture overview
3. Reference: **[FILES_SUMMARY.md](FILES_SUMMARY.md)** - What was changed

### For Integration
1. **[LLM_ANNOTATION_GUIDE.md](LLM_ANNOTATION_GUIDE.md)** - How to send feedback to LLM
2. **[backend/app/video_processing/ANALYSIS_README.md](backend/app/video_processing/ANALYSIS_README.md)** - Detailed technical docs
3. **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - Visual data flows

### For Developers
1. **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - System design
2. **[backend/app/video_processing/analysis.py](backend/app/video_processing/analysis.py)** - Source code (765 lines)
3. **[backend/app/video_processing/validate_analysis.py](backend/app/video_processing/validate_analysis.py)** - Validation script

---

## 📁 File Structure

```
trainer/
│
├─ QUICK_START.md ........................ Quick reference guide (TL;DR)
├─ IMPLEMENTATION_SUMMARY.md ............. Complete feature overview
├─ ARCHITECTURE_DIAGRAM.md .............. Visual system design & data flows
├─ LLM_ANNOTATION_GUIDE.md .............. Guide for LLM integration
├─ FILES_SUMMARY.md ..................... Summary of all changes
├─ README.md (this file)
│
└─ backend/app/video_processing/
   │
   ├─ analysis.py ....................... MAIN: 765 lines, fully featured
   ├─ ANALYSIS_README.md ................ Complete technical documentation
   ├─ validate_analysis.py .............. Quick validation script
   ├─ test_analysis.py .................. Test harness
   │
   └─ pose_outputs/
      ├─ biomechanical_data.csv ......... INPUT: Raw pose data from model
      ├─ rep_advanced_metrics.csv ....... OUTPUT: Frame-by-frame analysis
      ├─ comparison_results.csv ......... OUTPUT: Detailed comparison
      ├─ comparison_summary.csv ......... OUTPUT: Per-rep scores
      └─ comparison_visualization.png ... OUTPUT: Signal overlay plots
```

---

## 🚀 Getting Started in 3 Steps

### Step 1: Validate Installation
```bash
cd backend/app/video_processing
python validate_analysis.py
```
Expected output: ✓ ALL VALIDATION TESTS PASSED

### Step 2: Run Analysis
```bash
python analysis.py
```
Output: `pose_outputs/rep_advanced_metrics.csv`

### Step 3: Compare Two Videos
```python
from analysis import compare_exercises

detailed_df, summary_df = compare_exercises(
    'pose_outputs/ground_truth_metrics.csv',
    'pose_outputs/user_video_metrics.csv'
)
print(summary_df[['rep_id', 'form_quality_score', 'overall_score']])
```

---

## 📊 What Gets Generated

| File | Type | Contains | Rows |
|------|------|----------|------|
| `rep_advanced_metrics.csv` | Analysis | Frame-by-frame signals + rep summaries | ~300 |
| `comparison_results.csv` | Detailed | Frame data + annotations (when flagged) | ~300 |
| `comparison_summary.csv` | Summary | Per-rep quality scores & deviations | N reps |
| `comparison_visualization.png` | Chart | Signal overlay plots (GT vs rep) | N/A |

---

## 🔧 Configuration

All in `backend/app/video_processing/analysis.py` (top of file):

```python
# Tolerances (how much deviation triggers flagging)
SIGNAL_TOLERANCES = {
    'angle': 5.0,              # degrees
    'speed': 0.10,             # 10% of value
    'symmetry': 2.0,           # degrees
    # etc.
}

# Scoring weights
JOINT_WEIGHT = 0.70            # 70% for joint mechanics
SYMMETRY_WEIGHT = 0.30         # 30% for bilateral balance
```

---

## 🎯 Key Features

### Multi-Rep Support
- ✅ Automatic detection of multiple reps
- ✅ Each rep gets unique `rep_id` (0, 1, 2, ...)
- ✅ Individual max_depth_frame per rep
- ✅ Rest periods calculated between reps

### Ground-Truth Comparison
- ✅ Dynamic time-warping of GT signals
- ✅ Rep duration difference tracked
- ✅ Tolerance-based deviation flagging
- ✅ Metric-only annotations for LLM

### Comprehensive Analysis
- ✅ 150+ signals per frame (angles, speeds, symmetries)
- ✅ 33 keypoints tracked
- ✅ 8 joint angles extracted
- ✅ 5 symmetry scores calculated

### Smart Scoring
- ✅ Form quality: 70% weight on joints
- ✅ Symmetry quality: 30% weight
- ✅ Combined overall score: 0-100%
- ✅ Primary deviations identified (top 3)

---

## 📖 Documentation Map

```
QUICK START
    ↓
    ├─→ Want visual overview? Read ARCHITECTURE_DIAGRAM.md
    ├─→ Want details? Read IMPLEMENTATION_SUMMARY.md
    ├─→ Want LLM integration? Read LLM_ANNOTATION_GUIDE.md
    └─→ Need references? Read ANALYSIS_README.md

ARCHITECTURE_DIAGRAM
    ├─→ System pipeline diagram
    ├─→ Data flow visualization
    ├─→ CSV structure examples
    ├─→ Time-warping explanation
    └─→ Tolerance & scoring formula

IMPLEMENTATION_SUMMARY
    ├─→ Features implemented
    ├─→ Output file specifications
    ├─→ Signal types reference
    ├─→ Usage examples
    └─→ Configuration guide

LLM_ANNOTATION_GUIDE
    ├─→ Annotation format spec
    ├─→ Extraction methods
    ├─→ Prompt templates
    ├─→ Parsing code
    └─→ Example LLM output

ANALYSIS_README
    └─→ Complete technical documentation
```

---

## 🔍 Understanding the Outputs

### rep_advanced_metrics.csv
Frame-by-frame data with rep metadata:
- `rep_id`: Identifies which rep each frame belongs to (0, 1, 2, ... or NaN for rest)
- Signals: 150+ columns (angles, speeds, symmetries, etc.)
- Rep summaries: start_frame, end_frame, max_depth_frame, duration, etc.

**Key insight:** Every frame contains full rep metadata for easy filtering

### comparison_results.csv
Detailed comparison with annotations:
- Full frame data from comparison rep
- `{signal}_annotation` columns populated ONLY on flagged frames
- Format: `"joint_name|gt_value|actual_value|difference"`
- Example: `"left_elbow|89.5|76.2|13.3"` (13.3° off target)

### comparison_summary.csv
Per-rep scores:
- `form_quality_score`: 0-100% (joint metrics weighted 70%)
- `symmetry_quality_score`: 0-100% (symmetry metrics weighted 30%)
- `overall_score`: Weighted combination
- `primary_joint_deviations`: Top 3 problem joints
- `rep_duration_diff_ms`: How much slower/faster than GT

---

## 💡 Common Tasks

### Find worst deviations
```python
import pandas as pd
df = pd.read_csv('comparison_results.csv')
# Get top 10 frames with most flagged signals
df['num_issues'] = df.filter(like='_annotation').notna().sum(axis=1)
worst = df.nlargest(10, 'num_issues')
```

### Compare multiple attempts
```python
for video in ['attempt1', 'attempt2', 'attempt3']:
    _, summary = compare_exercises('gt_metrics.csv', f'{video}_metrics.csv')
    print(f"{video}: {summary.iloc[0]['overall_score']:.1f}%")
```

### Extract for LLM processing
```python
detailed = pd.read_csv('comparison_results.csv')
summary = pd.read_csv('comparison_summary.csv')

# Build prompt
prompt = f"Rep 1 score: {summary.iloc[0]['overall_score']:.0f}%\n"
prompt += f"Issues: {summary.iloc[0]['primary_joint_deviations']}\n"
# Send to LLM
```

### Analyze specific joints
```python
df = pd.read_csv('comparison_results.csv')
# Get all left_elbow deviations
left_elbow = df[df['left_elbow_annotation'].notna()]
print(f"Left elbow issues: {len(left_elbow)} frames")
```

---

## ⚙️ Customization

### Change tolerance thresholds
Edit `analysis.py` line ~20:
```python
SIGNAL_TOLERANCES = {
    'angle': 3.0,   # Stricter (was 5.0)
    'speed': 0.15,  # More lenient (was 0.10)
}
```

### Adjust scoring weights
Edit `analysis.py` line ~28:
```python
JOINT_WEIGHT = 0.80        # Prioritize joints (was 0.70)
SYMMETRY_WEIGHT = 0.20     # Less symmetry weight (was 0.30)
```

### Modify rep detection sensitivity
In `detect_rep_points()` function:
```python
tolerance = 0.005  # Stricter (was 0.01)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No reps detected | Check elbow displacement has variation; adjust `tolerance` in `detect_rep_points()` |
| All frames flagged | Tolerances too strict; increase SIGNAL_TOLERANCES values |
| NaN values in output | Low confidence frames; check CONFIDENCE_THRESHOLD if needed |
| Large annotation file | Many deviations; compare is very different from ground truth |
| Validation fails | Check all imports: pandas, numpy, scipy, matplotlib |

---

## 📚 Document Index

| Document | Purpose | Audience | Length |
|----------|---------|----------|--------|
| QUICK_START.md | Quick reference | All users | 150 lines |
| IMPLEMENTATION_SUMMARY.md | Feature overview | Developers | 400 lines |
| ARCHITECTURE_DIAGRAM.md | Visual design | Architects | 500 lines |
| LLM_ANNOTATION_GUIDE.md | LLM integration | Integration engineers | 300 lines |
| ANALYSIS_README.md | Technical docs | Developers | 300 lines |
| FILES_SUMMARY.md | Change summary | Project managers | 200 lines |
| backend/app/video_processing/analysis.py | Source code | Developers | 765 lines |

---

## ✅ Validation Checklist

- [x] All imports available (pandas, numpy, scipy, matplotlib)
- [x] Signal warping function tested
- [x] Multi-rep detection verified
- [x] Time-warping algorithm reviewed
- [x] Tolerance-based flagging working
- [x] Annotation format correct
- [x] Score calculation validated
- [x] CSV output structures verified
- [x] Rest period calculation confirmed
- [x] Documentation complete (2000+ lines)

---

## 🎓 Key Concepts

**Rep Detection:** Uses elbow Y-displacement peaks to automatically identify start, end, and max_depth for each rep

**Time-Warping:** Ground truth signals are dynamically stretched/compressed to match each rep's duration for fair comparison

**Tolerance Thresholds:** Per-signal limits (e.g., angle ±5°, speed ±10%) trigger flagging when exceeded

**Metric-Only Annotations:** Clean format "joint|gt|actual|diff" designed for LLM processing

**Weighted Scoring:** Joints 70% weight, Symmetry 30% weight in overall quality calculation

**Rest Periods:** Automatically identified as gaps between reps (marked with NaN rep_id)

---

## 🚦 Next Steps

1. Read **QUICK_START.md** (5 min)
2. Run `validate_analysis.py` (1 min)
3. Run `analysis.py` on sample data (2 min)
4. Review output CSVs (5 min)
5. Run comparison on two videos (2 min)
6. Read **LLM_ANNOTATION_GUIDE.md** (10 min)
7. Test LLM integration (10 min)
8. Fine-tune tolerances based on real data

Total time to full implementation: ~45 minutes

---

**Status: ✅ COMPLETE AND READY TO USE**

All documentation, code, and examples are provided. System is production-ready.

For questions or issues, refer to the relevant documentation or examine the source code in `analysis.py`.