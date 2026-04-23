"""
LSTM ve RL feature pipeline arasında paylaşılan sabitler.

TIME_STEP değişirse hem lstm_model hem prepare_prediction_states güncellenmeli;
tek kaynak burası olsun.
"""

LSTM_TIME_STEP: int = 12
