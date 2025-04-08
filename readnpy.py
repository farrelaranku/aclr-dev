import numpy as np
import matplotlib.pyplot as plt

# Load data
pred_file = "checkpoints/weather_96_96_withACLR_b128_B6iFast_custom_ftM_sl96_ll48_pl96_dm512_el3_dl1_Exp/testing_results/mantra/pred.npy"
true_file = "checkpoints/weather_96_96_withACLR_b128_B6iFast_custom_ftM_sl96_ll48_pl96_dm512_el3_dl1_Exp/testing_results/mantra/true.npy"

pred_data = np.load(pred_file)
true_data = np.load(true_file)

# Pilih satu sampel
sample_index = 3  # Bisa diganti sesuai keinginan
feature_index = 18  # Pilih fitur yang ingin diplot (misal fitur ke-0)

pred_feature = pred_data[sample_index, :, feature_index]  # (336,)
true_feature = true_data[sample_index, :, feature_index]  # (336,)

# Plot prediksi dan ground truth
plt.figure(figsize=(12, 6))
plt.plot(pred_feature, label="Prediksi", linestyle="solid", color="blue")
plt.plot(true_feature, label="Ground Truth", linestyle="solid", color="red")
# plt.ylim(1, 10)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title(f"Prediksi vs Ground Truth (Sampel {sample_index}, Fitur {feature_index})")
plt.legend()

# Simpan sebagai PDF
pdf_filename = "perbandingan_pred_true_fitur_0.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")

# Tampilkan plot
plt.show()

print(f"Plot berhasil disimpan sebagai {pdf_filename}")
