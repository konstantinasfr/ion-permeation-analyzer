## ✅ 5. Radial Deviation from the Channel Axis

### 🎯 Purpose:
Track how far each ion deviates from the central axis of the channel over time. This helps identify when ions are:

- **Centered** → likely following the preferred conduction path.
- **Off-center** → possibly interacting with side chains, getting transiently trapped, or detouring before permeating.

### 📐 Method:

The radial distance `r` from the channel center is calculated as:

```python
r = sqrt((x - x0)² + (y - y0)²)
