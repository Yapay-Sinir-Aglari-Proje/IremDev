"""
Geriye dönük uyumluluk: kök dizinde `python evaluate_performance.py` çalıştırıldığında
sys.path ayarı yapılır ve asıl giriş noktası `evaluate.main` ile aynı CLI’yi kullanır.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import main

if __name__ == "__main__":
    main()
