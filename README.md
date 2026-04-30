# Note
因GITHUB無法上傳單一大於25MB之檔案，因此在result與data資料夾內容有疏漏，以E3檔案繳交提供。
# 本次模擬之五個不同場景
1. 場景一：全區隨機分佈 (Randomly distributed)
* 空間位置：使用者隨機出現於整個 $20\text{m} \times 20\text{m}$ 的區域內（$X$ 與 $Y$ 座標皆介於 $-10$ 到 $10$ 之間）。
* 移動狀態：以每秒 $0.5\text{m}$ 的速度，朝 $X$ 與 $Y$ 的正向斜角緩慢移動。
* 特徵意義：作為最基礎且多樣化的通道資料，涵蓋了各種距離與角度的訊號衰落。
本實驗比較了基於深度神經網路 (DNN) 與傳統線性最小均方誤差 (LMMSE) 的通道估測器 (Channel Estimator) 在 OFDM 系統中的均方誤差 (Mean Square Error, MSE) 表現。測試涵蓋了 5 dB 至 40 dB 的訊雜比 (SNR) 範圍，並針對「具備循環前綴 (with CP)」與「移除循環前綴 (without CP)」兩種情境進行了深度探討。
<p align="center">
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/803ccaf6-bede-4d72-a998-562649a9c868" />
</p>
<p align="center"> <strong>Figure 1</strong> </p>

## 根據 Figure 1 的模擬結果，我們可以觀察到以下幾個關鍵結論：
### 理想情境（具備 CP，無符號間干擾）：
* 在系統擁有完整 CP 的理想條件下（圖中實線），系統不存在符號間干擾 (Inter-Symbol Interference, ISI)。
* **LMMSE 估測器**（藍色空心圓，LMMSE (with CP)）在此線性條件下展現了最佳的基準性能，其 MSE 隨著 SNR 的提升而穩定且大幅地下降（在 40 dB 時達到約 $2 \times 10^{-5}$）。
* **DNN 估測器**（紅色空心方塊，DNN (with CP)）的曲線與 LMMSE 高度貼合。這證明了在理想通道條件下，數據驅動 (Data-Driven) 的神經網路模型具備足夠的學習能力，能夠完美擬合並重現傳統最佳線性估測演算法的表現。
### 非理想嚴苛情境（移除 CP，引發嚴重 ISI）：
* 當移除 CP 時（圖中虛線），相鄰 OFDM 符號間會產生嚴重的 ISI，使得通道模型呈現高度的非線性與干擾。
* **傳統算法的崩潰 (Error Floor)**：傳統的 LMMSE 估測器（藍色實心圓，LMMSE (without CP)）完全無法處理這類非線性干擾。無論 SNR 如何提升，其 MSE 始終停滯在極高的誤差底線（約在 $4 \times 10^{-2}$ 附近），完全失去通道估測的有效性。
* **DNN 的強健性 (Robustness)**：與之形成強烈對比的是，DNN 估測器（紅色實心方塊，DNN (without CP)）展現了極佳的抗干擾能力。即使在缺乏 CP 的情況下，其 MSE 依然能隨著 SNR 的增加而持續下降，軌跡僅略微高於理想的 CP 情境。這顯示 DNN 能夠透過其隱藏層成功學習到 ISI 的干擾模式，並在估測過程中進行了有效的非線性補償。
# 總結 (Conclusion)
這項實驗證明了資料驅動通道估測器的巨大潛力。在常規條件下，DNN 能夠達到媲美 LMMSE 的精確度；而在面對嚴苛的物理限制與複雜的通道失真（如缺乏 CP 導致的 ISI）時，DNN 則能突破傳統線性演算法的理論瓶頸，維持高度可靠的通訊品質。
