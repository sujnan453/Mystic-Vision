# Auto-Brightness Feature - Quick Guide

## ✨ What It Does

Automatically analyzes your camera feed and adjusts brightness for optimal hand detection!

---

## 🎮 How to Use

### Toggle Auto-Brightness:
**Press 'A' key** to turn ON/OFF

### Visual Indicator:
- **Green button** = AUTO mode (enabled)
- **Gray button** = MANUAL mode (disabled)
- Shows `[A] AUTO` or `[A] MANUAL` in top-left corner

### Brightness Display:
- Shows current brightness value
- Adds "(AUTO)" when auto-brightness is active
- Example: `Brightness: 45 (AUTO)`

---

## 🔧 How It Works

### Analysis:
1. Converts frame to grayscale
2. Calculates average brightness (0-255)
3. Determines optimal brightness boost

### Brightness Levels:
- **Very Dark** (< 80): Boost = 60
- **Dark** (80-110): Boost = 45
- **Slightly Dark** (110-140): Boost = 30
- **Good** (140-170): Boost = 15
- **Bright** (> 170): Boost = 0

### Smart Updates:
- Updates every 2 seconds (prevents flickering)
- Smooth transitions (70% old + 30% new)
- Automatic adjustment based on lighting conditions

---

## 📍 UI Location

```
┌────────────────────────────────┐
│ [A] AUTO  ← Button (top-left)  │
│ Brightness: 45 (AUTO)          │
│ Detection: 85%                 │
│ [████████░░]                   │
└────────────────────────────────┘
```

---

## 🎯 Benefits

✅ **Adapts to lighting** - Works in any environment  
✅ **Improves detection** - Optimal brightness for hand tracking  
✅ **No manual adjustment** - Set it and forget it  
✅ **Smooth transitions** - No sudden brightness jumps  
✅ **Exhibition-ready** - Perfect for changing lighting conditions  

---

## 🚀 Recommended Usage

### For Exhibitions:
1. Start application
2. Press 'A' to enable auto-brightness
3. Let it analyze for 2-4 seconds
4. Brightness will auto-adjust
5. Leave it ON for the entire exhibition

### For Testing:
1. Test in different lighting (bright, dark, mixed)
2. Watch brightness value adjust automatically
3. Verify hand detection improves
4. Toggle OFF if you prefer manual control

---

## ⌨️ Keyboard Shortcut

**A** = Toggle Auto-Brightness ON/OFF

---

## 🎪 Exhibition Setup

**Recommended Command:**
```bash
python shield.py --camera 1 --demo --start_fullscreen
```

**Then:**
1. Press 'A' to enable auto-brightness
2. Wait 2-4 seconds for initial adjustment
3. Ready for visitors!

---

**Perfect for exhibitions with varying lighting conditions! 🌟**
