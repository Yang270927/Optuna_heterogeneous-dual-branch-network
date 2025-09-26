import matplotlib.pyplot as plt

# optuna 0.6695
y1 = [0.1524,0.1625,0.1822,0.2170,0.3271,0.3960,0.5436,0.7068,0.7803,0.8458,0.8939,
      0.9090,0.9065,0.9086,0.9051,0.9059,0.9200,0.9017,0.9142,0.9122]
average1 = sum(y1) / len(y1)
print(f"均值为: {average1:.4f}")

#DAE 0.5614
y2 = [0.0904,0.0950,0.0990,0.1259,0.1609,0.2240,0.3600,0.5104,0.6000,0.6950,0.7636,
      0.8040,0.8377,0.8309,0.8345,0.8440,0.8418,0.8331,0.8350,0.8431]
average2 = sum(y2) / len(y2)
print(f"均值为: {average2:.4f}")
#MCLDNN 0.6154
y3 = [0.0945,0.0922,0.0945,0.1222,0.1659,0.2354,0.3668,0.5368,0.6554,0.8027,0.8809,
      0.8986,0.9159,0.9131,0.9236,0.9213,0.9263,0.9154,0.9200,0.9263]
average3 = sum(y3) / len(y3)
print(f"均值为: {average3:.4f}")
#PET_CGDNN 0.6039
y4 = [0.0954,0.0954,0.1000,0.1331,0.1840,0.2645,0.3722,0.5077,0.6400,0.7609,0.8481,
      0.8777,0.8972,0.8972,0.8972,0.9077,0.9022,0.8972,0.9004,0.9004]
average4 = sum(y4) / len(y4)
print(f"均值为: {average4:.4f}")
#MHNN 0.6218
y5 = [0.0955,0.1017,0.1130,0.1555,0.1833,0.3016,0.3929,0.5788,0.7284,0.8129,0.8518,
      0.8778,0.8897,0.9098,0.8995,0.9159,0.9101,0.9128,0.9067,0.9000]
average5 = sum(y5) / len(y5)
print(f"均值为: {average5:.4f}")
#WACN 0.5709
y6 = [0.0999,0.0971,0.1031,0.1359,0.1578,0.2318,0.3636,0.5120,0.6584,0.7273,0.7824]
      # 0.8312,0.8274,0.8389,0.8285,0.8447,0.8463,0.8443,0.8460,0.8408]
average6 = sum(y6) / len(y6)
print(f"均值为: {average6:.4f}")
# optuna_BN 0.6555
# y7 = [0.1295,0.1366,0.1673,0.2162,0.2892,0.3432,0.4835,0.6634,0.7487,0.8373,0.8876,0.9003,0.9004,0.9019,0.9080,0.9078,0.9149,0.9174,0.9170,0.9203]

# optuna_BN_Z-score 0.6730
# y8 = [0.1588,0.1652,0.1872,0.2413,0.3239,0.4017,0.5208,0.7207,0.7938,0.8625,0.9057,0.9027,0.9042,0.9088,0.9071,0.9168,0.9153,0.9120,0.9146,0.9218]

x = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
plt.figure(figsize=(6, 6))

plt.plot(x, y1, linewidth=1, color="red", marker="o")
plt.plot(x, y2, linewidth=1, color="orange", marker="o")
plt.plot(x, y3, linewidth=1, color="blue", marker="o")
plt.plot(x, y4, linewidth=1, color="green", marker="o")
plt.plot(x, y5, linewidth=1, color="purple", marker="o")
plt.plot(x, y6, linewidth=1, color="cyan", marker="o")
# plt.plot(x, y7, linewidth=1, color="pink", marker="o")
# plt.plot(x, y8, linewidth=1, color="brown", marker="o")


plt.xlabel("SNR(dB)")
plt.ylabel("accuracy")
# plt.title(f'recognition accuracy of RML2016.10a')

plt.legend(["optuna","DAE","MCLDNN","PET_CGDNN","MHNN","WACN"], loc="lower right")

plt.xlim([-20, 20])
plt.ylim([0, 1])

plt.grid()
acc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.yticks(acc)
plt.xticks(x)

plt.show()

